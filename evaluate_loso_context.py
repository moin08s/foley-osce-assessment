"""LOSO Cross-Validation with Neighbouring-Window Context (Task 5).

Replaces the independent per-window MLP with temporal window context:
Each window t is represented by [x_{t-1}, x_t, x_{t+1}] (stacked neighbouring-window context),
capturing procedural transitions, velocity, and local temporal momentum.

Pure Python + NumPy ONLY.
Evaluates 24-fold LOSO cross-validation on 128x128 4fps features.
"""

import os
import glob
import json
import time
from typing import Dict, List, Tuple
import numpy as np

from src.model import FoleyStepClassifier
from src.session_checker import SessionChecker
from src.grading import OSCEGrader
from src.data_loader import STEP_LABELS


def create_context_features(X: np.ndarray, window_radius: int = 1) -> np.ndarray:
    """Stack neighbouring window features [x_{t-K}, ..., x_t, ..., x_{t+K}].
    
    Args:
        X: (T, D) sequence of window features
        window_radius: K, number of past and future windows to stack (K=1 -> 3 windows)
        
    Returns:
        (T, (2K+1)*D) contextualized feature matrix
    """
    T, D = X.shape
    if T == 0:
        return np.zeros((0, (2 * window_radius + 1) * D), dtype=np.float32)

    # Pad at edges with replication
    padded = np.pad(X, ((window_radius, window_radius), (0, 0)), mode="edge")
    
    context_list = []
    for offset in range(2 * window_radius + 1):
        context_list.append(padded[offset : offset + T])
        
    # Stack along feature axis
    X_context = np.hstack(context_list).astype(np.float32)
    return X_context


def load_context_sessions(features_dir: str, gt_path: str, radius: int = 1):
    """Load 128x128 4fps features, apply ablation, per-video norm, and context stacking."""
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

        # Stack neighbouring windows
        X_ctx = create_context_features(X_norm, window_radius=radius)

        sessions[stem] = {
            "stem": stem,
            "X": X_ctx,
            "y": y,
            "gt": cohort_gt[stem],
        }

    return sessions


def augment_context_features(X: np.ndarray, y: np.ndarray, radius: int = 1, base_dim: int = 163) -> Tuple[np.ndarray, np.ndarray]:
    """Apply augmentations respecting context structure."""
    T, D = X.shape
    X_aug_list = [X]
    y_aug_list = [y]

    # Augmentation 1: Gaussian Noise + Color jitter across stacked windows
    X_noise = X.copy()
    X_noise += np.random.normal(0, 0.04, size=(T, D)).astype(np.float32)
    scale = np.random.uniform(0.85, 1.15)
    for k in range(2 * radius + 1):
        offset = k * base_dim
        X_noise[:, offset + 65 : offset + 117] *= scale
    X_aug_list.append(X_noise)
    y_aug_list.append(y)

    # Augmentation 2: Temporal Speed Warping
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


def train_context_model(
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
    """Train MLP with larger hidden dimension for context features."""
    input_dim = X_train.shape[1]
    model = FoleyStepClassifier(
        input_dim=input_dim,
        hidden_dim1=256,
        hidden_dim2=128,
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

    if best_weights is not None:
        model.W1 = best_weights["W1"]
        model.b1 = best_weights["b1"]
        model.W2 = best_weights["W2"]
        model.b2 = best_weights["b2"]
        model.W3 = best_weights["W3"]
        model.b3 = best_weights["b3"]

    return model


def run_context_loso(sessions: Dict[str, Dict], radius: int = 1):
    """Execute 24-fold LOSO with neighbouring-window context model."""
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
    print(f"TASK 5: NEIGHBOURING-WINDOW CONTEXT (RADIUS={radius}, 3-WINDOW STACK) LOSO ({n_folds} FOLDS)")
    print("=" * 95)
    print(f"{'FOLD':5s} | {'HELD-OUT':10s} | {'NAME':14s} | {'WIN ACC':8s} | {'TEACHER':8s} | {'MODEL':8s} | {'RESULT':8s} | {'SCORE':8s}")
    print("-" * 95)

    for i, held_out_stem in enumerate(stems):
        train_pool_stems = [s for s in stems if s != held_out_stem]

        # Inner validation split (4 val, 19 train)
        val_stems = [train_pool_stems[(i + k) % len(train_pool_stems)] for k in range(4)]
        train_stems = [s for s in train_pool_stems if s not in val_stems]

        # Augment train sessions
        X_train_list = []
        y_train_list = []
        for s in train_stems:
            x_aug, y_aug = augment_context_features(sessions[s]["X"], sessions[s]["y"], radius=radius)
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

        # Train model with context
        model = train_context_model(
            X_train, y_train,
            X_val, y_val,
            max_epochs=150,
            patience=20,
            lr=3e-3,
            seed=i + 300,
        )

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

    # Per-step precision & recall
    per_step_metrics = {}
    print("\n" + "=" * 95)
    print("PER-STEP WINDOW CLASSIFICATION METRICS (HELD-OUT SESSIONS WITH CONTEXT)")
    print("=" * 95)
    print(f"{'STEP':4s} | {'STEP DESCRIPTION':32s} | {'PRECISION':10s} | {'RECALL':10s} | {'F1':10s} | {'SUPPORT':8s}")
    print("-" * 95)

    for c in range(14):
        step_name = STEP_LABELS.get(c, "Background")
        tp = int(np.sum((y_pred_arr == c) & (y_true_arr == c)))
        fp = int(np.sum((y_pred_arr == c) & (y_true_arr != c)))
        fn = int(np.sum((y_pred_arr != c) & (y_true_arr == c)))
        support = int(np.sum(y_true_arr == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_step_metrics[c] = {
            "name": step_name,
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "support": support,
        }
        print(f"{c:4d} | {step_name:32s} | {prec*100:9.1f}% | {rec*100:9.1f}% | {f1*100:9.1f}% | {support:8d}")

    # Detection sensitivity
    print("\n" + "=" * 95)
    print("SEGMENT-LEVEL STEP DETECTION SENSITIVITY (HELD-OUT SESSIONS)")
    print("=" * 95)
    print(f"{'STEP':4s} | {'STEP DESCRIPTION':32s} | {'PRESENT':8s} | {'DETECTED':8s} | {'SENSITIVITY':12s}")
    print("-" * 95)
    for s in range(1, 14):
        pres = step_detection_counts[s]["present"]
        det = step_detection_counts[s]["detected"]
        sens = (det / pres) * 100.0 if pres > 0 else 0.0
        step_detection_counts[s]["sensitivity_pct"] = round(sens, 1)
        step_name = STEP_LABELS.get(s, "")
        print(f"{s:4d} | {step_name:32s} | {pres:8d} | {det:8d} | {sens:11.1f}%")

    # Load baseline, ablation, augmented, 128px for comparison
    with open(r"C:\Medical\reports\loso_baseline_report.json") as f:
        baseline = json.load(f)
    with open(r"C:\Medical\reports\loso_128px_report.json") as f:
        t4_128px = json.load(f)

    b_acc = baseline["overall_window_accuracy"]
    b_grd = baseline["final_grade_agreement_pct"]
    t4_acc = t4_128px["overall_window_accuracy"]
    t4_grd = t4_128px["final_grade_agreement_pct"]

    print("\n" + "=" * 95)
    print(f"TASK 5 PROGRESS COMPARISON SUMMARY ({elapsed:.1f}s total)")
    print("=" * 95)
    print(f"Task 1 (64px, 2fps, Baseline):       Window Acc: {b_acc:.2f}% | Final Grade Agreement: {b_grd:.1f}%")
    print(f"Task 4 (128px, 4fps, Independent):   Window Acc: {t4_acc:.2f}% | Final Grade Agreement: {t4_grd:.1f}%")
    print(f"Task 5 (128px, 4fps, Context Model): Window Acc: {overall_win_acc:.2f}% | Final Grade Agreement: {grade_agreement_pct:.1f}%")
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
        "comparison": {
            "task1_baseline_window_acc": b_acc,
            "task4_128px_window_acc": t4_acc,
            "task5_context_window_acc": round(overall_win_acc, 2),
            "delta_from_baseline": round(overall_win_acc - b_acc, 2),
            "delta_from_task4": round(overall_win_acc - t4_acc, 2),
        },
        "per_step_metrics": per_step_metrics,
        "step_detection_counts": step_detection_counts,
        "grade_confusion": grade_confusion,
        "session_results": session_results,
    }

    report_path = r"C:\Medical\reports\loso_context_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved detailed LOSO context report to: {report_path}")

    return out_json


if __name__ == "__main__":
    sessions = load_context_sessions(
        features_dir=r"C:\Medical\dataset\features_128px",
        gt_path=r"C:\Medical\dataset\cohort_ground_truth.json",
        radius=1,
    )
    run_context_loso(sessions, radius=1)
