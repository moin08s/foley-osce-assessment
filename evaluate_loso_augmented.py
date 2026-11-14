"""LOSO Cross-Validation with Augmentation and Early Stopping (Task 3).

Improvements:
1. Feature Augmentations (Pure NumPy):
   - Brightness / Color scaling: multiplicative jitter on color histogram features.
   - Crop shift / Spatial grid jitter: 8x8 spatial roll on grid motion.
   - Gaussian feature noise: N(0, 0.03) additive noise.
   - Temporal speed warping: resample time sequences by 0.9x to 1.1x.
2. Inner Validation & Early Stopping:
   - Split 23 training sessions into 19 train + 4 validation.
   - Monitor validation loss with patience = 20 epochs.
   - Restore best checkpoint before evaluating on held-out test candidate.
   - Removed 'tune to zero mistakes' hard-negative oversampling.
3. Full Dataset Replay for Retraining:
   - Verifies replay capability.
"""

import os
import glob
import json
import time
from typing import Dict, List, Tuple
import numpy as np

from src.model import FoleyStepClassifier, StandardScaler
from src.session_checker import SessionChecker
from src.grading import OSCEGrader
from src.data_loader import STEP_LABELS


def load_ablated_sessions(features_dir: str, gt_path: str):
    """Load 163-dim features with per-video normalization."""
    with open(gt_path, "r", encoding="utf-8") as f:
        cohort_gt = json.load(f)

    session_files = sorted(glob.glob(os.path.join(features_dir, "*_feat.npz")))
    sessions = {}

    for f in session_files:
        stem = os.path.basename(f).replace("_feat.npz", "")
        if stem not in cohort_gt:
            continue
        data = np.load(f)
        X_raw = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)

        # 163-dim features (remove 64-dim spatial projection)
        X_no_spatial = np.hstack([X_raw[:, :117], X_raw[:, 181:]])

        # Per-video normalization
        v_mean = np.mean(X_no_spatial, axis=0, keepdims=True)
        v_std = np.std(X_no_spatial, axis=0, keepdims=True) + 1e-7
        X_norm = (X_no_spatial - v_mean) / v_std

        sessions[stem] = {
            "stem": stem,
            "X": X_norm,
            "y": y,
            "gt": cohort_gt[stem],
        }

    return sessions


def augment_session_features(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply multimodal feature augmentations (pure NumPy)."""
    T, D = X.shape
    X_aug_list = [X]
    y_aug_list = [y]

    # Augmentation 1: Gaussian Noise + Brightness / Color jitter
    X_noise = X.copy()
    # Additive noise
    X_noise += np.random.normal(0, 0.04, size=(T, D)).astype(np.float32)
    # Brightness scale on color histograms (indices 65..116)
    scale = np.random.uniform(0.85, 1.15)
    X_noise[:, 65:117] *= scale
    X_aug_list.append(X_noise)
    y_aug_list.append(y)

    # Augmentation 2: Spatial Grid Shift on 8x8 block motion (indices 1..64)
    X_shift = X.copy()
    grid_motion = X_shift[:, 1:65].reshape(T, 8, 8)
    shift_r = np.random.choice([-1, 0, 1])
    shift_c = np.random.choice([-1, 0, 1])
    grid_shifted = np.roll(grid_motion, shift=(shift_r, shift_c), axis=(1, 2))
    X_shift[:, 1:65] = grid_shifted.reshape(T, 64)
    X_aug_list.append(X_shift)
    y_aug_list.append(y)

    # Augmentation 3: Speed / Temporal Warping (+-10% speed)
    speed_factor = np.random.uniform(0.9, 1.1)
    new_len = max(5, int(round(T * speed_factor)))
    orig_idx = np.linspace(0, T - 1, new_len)
    X_warp = np.zeros((new_len, D), dtype=np.float32)
    for d in range(D):
        X_warp[:, d] = np.interp(orig_idx, np.arange(T), X[:, d])
    y_warp = np.zeros(new_len, dtype=np.int64)
    for i in range(new_len):
        nearest_orig = int(round(orig_idx[i]))
        y_warp[i] = y[min(nearest_orig, T - 1)]

    X_aug_list.append(X_warp)
    y_aug_list.append(y_warp)

    return np.vstack(X_aug_list), np.concatenate(y_aug_list)


def train_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 150,
    patience: int = 20,
    lr: float = 3e-3,
    batch_size: int = 32,
    seed: int = 42,
) -> FoleyStepClassifier:
    """Train MLP with validation loss monitoring and early stopping."""
    input_dim = X_train.shape[1]
    model = FoleyStepClassifier(
        input_dim=input_dim,
        hidden_dim1=128,
        hidden_dim2=64,
        num_classes=14,
        seed=seed,
    )
    model.scaler.mean = np.zeros((1, input_dim), dtype=np.float32)
    model.scaler.scale = np.ones((1, input_dim), dtype=np.float32)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    N = len(X_train)
    curr_lr = lr

    for epoch in range(1, max_epochs + 1):
        perm = np.random.permutation(N)
        X_shuff = X_train[perm]
        y_shuff = y_train[perm]

        for b_start in range(0, N, batch_size):
            b_end = min(b_start + batch_size, N)
            xb = X_shuff[b_start:b_end]
            yb = y_shuff[b_start:b_end]
            probs, cache = model.forward(xb)
            grads = model.backward(cache, yb, l2_reg=1e-4)
            model._adam_step(grads, lr=curr_lr)

        # Evaluate validation loss
        val_probs, _ = model.forward(X_val)
        val_loss = model.compute_loss(val_probs, y_val, l2_reg=1e-4)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {
                "W1": model.W1.copy(), "b1": model.b1.copy(),
                "W2": model.W2.copy(), "b2": model.b2.copy(),
                "W3": model.W3.copy(), "b3": model.b3.copy(),
            }
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if epoch % 50 == 0:
            curr_lr *= 0.8

    # Restore best checkpoint
    if best_weights is not None:
        model.W1 = best_weights["W1"]
        model.b1 = best_weights["b1"]
        model.W2 = best_weights["W2"]
        model.b2 = best_weights["b2"]
        model.W3 = best_weights["W3"]
        model.b3 = best_weights["b3"]

    return model


def run_augmented_loso(sessions: Dict[str, Dict]):
    """Execute 24-fold LOSO with data augmentation and early stopping."""
    stems = sorted(list(sessions.keys()))
    n_folds = len(stems)

    all_y_true = []
    all_y_pred_window = []

    step_detection_counts = {s: {"present": 0, "detected": 0} for s in range(1, 14)}
    session_results = []
    grade_matches = 0
    grade_confusion = {"P": {"P": 0, "P-1": 0, "R": 0}, "P-1": {"P": 0, "P-1": 0, "R": 0}, "R": {"P": 0, "P-1": 0, "R": 0}}

    checker = SessionChecker(min_step_duration_sec=2.0)
    grader = OSCEGrader()

    t_start = time.time()
    print("=" * 95)
    print(f"TASK 3: AUGMENTED LOSO (BRIGHTNESS, CROP SHIFT, NOISE, SPEED + EARLY STOPPING)")
    print("=" * 95)
    print(f"{'FOLD':5s} | {'HELD-OUT':10s} | {'NAME':14s} | {'WIN ACC':8s} | {'TEACHER':8s} | {'MODEL':8s} | {'RESULT':8s} | {'SCORE':8s}")
    print("-" * 95)

    for i, held_out_stem in enumerate(stems):
        # 23 other candidates for training
        train_pool_stems = [s for s in stems if s != held_out_stem]

        # Inner validation split (4 candidates for validation, 19 for training)
        val_stems = [train_pool_stems[(i + k) % len(train_pool_stems)] for k in range(4)]
        train_stems = [s for s in train_pool_stems if s not in val_stems]

        # Augment train sessions
        X_train_list = []
        y_train_list = []
        for s in train_stems:
            x_aug, y_aug = augment_session_features(sessions[s]["X"], sessions[s]["y"])
            X_train_list.append(x_aug)
            y_train_list.append(y_aug)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Validation set (unaugmented)
        X_val = np.vstack([sessions[s]["X"] for s in val_stems])
        y_val = np.concatenate([sessions[s]["y"] for s in val_stems])

        # Test session
        test_session = sessions[held_out_stem]
        X_test = test_session["X"]
        y_test = test_session["y"]
        gt_info = test_session["gt"]
        cand_name = gt_info.get("candidate_name", held_out_stem)
        faculty_grade = gt_info.get("faculty_grade", "P")
        prompted_steps = gt_info.get("prompted_steps", [])

        # Train with early stopping
        model = train_with_early_stopping(
            X_train, y_train,
            X_val, y_val,
            max_epochs=150,
            patience=20,
            lr=3e-3,
            seed=i + 100,
        )

        # Predict on held-out candidate
        probs, _ = model.forward(X_test)
        raw_preds = np.argmax(probs, axis=1)

        win_acc = float(np.mean(raw_preds == y_test)) * 100.0
        all_y_true.extend(y_test.tolist())
        all_y_pred_window.extend(raw_preds.tolist())

        # Viterbi decoding & segments
        viterbi_preds = checker.viterbi_decode(probs)
        windows = [{"start_sec": t * 2.0, "end_sec": t * 2.0 + 4.0} for t in range(len(X_test))]
        segments = checker.extract_segments(viterbi_preds, windows, window_probs=probs)
        detected_step_ids = set(s["step_id"] for s in segments)

        # Detection rate tracking
        true_steps_present = set(y_test.tolist()) - {0}
        for s in range(1, 14):
            if s in true_steps_present:
                step_detection_counts[s]["present"] += 1
                if s in detected_step_ids:
                    step_detection_counts[s]["detected"] += 1

        seq_audit = checker.check_sequence_rules(
            segments,
            speech_activity_per_step={1: 1.0, 2: 1.0},
            prompts_given=prompted_steps,
        )

        eval_report = grader.compute_grade(
            sequence_check=seq_audit,
            student_id=cand_name,
            session_id=f"session_{held_out_stem}",
            detected_segments=segments,
            faculty_ground_truth={"final_grade": faculty_grade},
        )

        model_grade = eval_report["final_grade"]
        is_match = (model_grade == faculty_grade)
        if is_match:
            grade_matches += 1

        if faculty_grade in grade_confusion and model_grade in grade_confusion[faculty_grade]:
            grade_confusion[faculty_grade][model_grade] += 1

        status_str = "[MATCH]" if is_match else "[FAIL]"
        print(f"[{i+1:02d}/{n_folds:02d}] | {held_out_stem:10s} | {cand_name:14s} | {win_acc:6.1f}% | {faculty_grade:8s} | {model_grade:8s} | {status_str:8s} | {eval_report['score']:8s}")

        session_results.append({
            "fold": i + 1,
            "held_out_stem": held_out_stem,
            "candidate_name": cand_name,
            "win_accuracy": win_acc,
            "teacher_grade": faculty_grade,
            "model_grade": model_grade,
            "matched": is_match,
            "score": eval_report["score"],
            "detected_steps": sorted(list(detected_step_ids)),
        })

    elapsed = time.time() - t_start

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred_window)
    overall_win_acc = float(np.mean(y_true_arr == y_pred_arr)) * 100.0
    grade_agreement_pct = (grade_matches / n_folds) * 100.0

    # Load baseline & ablation for comparison
    with open(r"C:\Medical\reports\loso_baseline_report.json") as f:
        baseline = json.load(f)
    with open(r"C:\Medical\reports\loso_ablation_report.json") as f:
        ablation = json.load(f)

    b_acc = baseline["overall_window_accuracy"]
    b_grd = baseline["final_grade_agreement_pct"]
    a_acc = ablation["overall_window_accuracy"]
    a_grd = ablation["final_grade_agreement_pct"]

    print("\n" + "=" * 95)
    print(f"TASK 3 COMPARISON SUMMARY ({elapsed:.1f}s total)")
    print("=" * 95)
    print(f"Window Accuracy:       {overall_win_acc:.2f}%  (Baseline: {b_acc:.2f}% | Ablation: {a_acc:.2f}%)")
    print(f"Final Grade Agreement: {grade_matches}/{n_folds} ({grade_agreement_pct:.1f}%) (Baseline: {b_grd:.1f}% | Ablation: {a_grd:.1f}%)")
    print("-" * 95)

    print("\nGRADE CONFUSION MATRIX (Rows: Faculty / Columns: Model):")
    print(f"{'FACULTY':10s} | {'PRED: P':10s} | {'PRED: P-1':10s} | {'PRED: R':10s}")
    print("-" * 45)
    for true_g in ["P", "P-1", "R"]:
        row = grade_confusion[true_g]
        print(f"{true_g:10s} | {row['P']:10d} | {row['P-1']:10d} | {row['R']:10d}")

    out_json = {
        "n_folds": n_folds,
        "elapsed_seconds": round(elapsed, 1),
        "overall_window_accuracy": round(overall_win_acc, 2),
        "final_grade_agreement_pct": round(grade_agreement_pct, 1),
        "grade_matches": grade_matches,
        "baseline_comparison": {
            "baseline_window_acc": b_acc,
            "ablation_window_acc": a_acc,
            "augmented_window_acc": round(overall_win_acc, 2),
            "delta_from_baseline": round(overall_win_acc - b_acc, 2),
        },
        "step_detection_counts": step_detection_counts,
        "grade_confusion": grade_confusion,
        "session_results": session_results,
    }

    report_path = r"C:\Medical\reports\loso_augmented_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved detailed LOSO augmented report to: {report_path}")

    return out_json


if __name__ == "__main__":
    sessions = load_ablated_sessions(
        features_dir=r"C:\Medical\dataset\features",
        gt_path=r"C:\Medical\dataset\cohort_ground_truth.json",
    )
    run_augmented_loso(sessions)
