"""LOSO Cross-Validation with Confidence and Out-of-Distribution (OOD) Check (Task 6).

Implements:
1. Predictive Confidence Estimation:
   - Maximum softmax probability per window m_t = max_c P_{t,c}
   - Session Mean Confidence: \bar{m}
   - Ambiguous Window Ratio: fraction of windows with max probability < 0.35
   - Mean Shannon Entropy across the session: H = -sum(p * log p)
2. Out-of-Distribution (OOD) Detection:
   - Feature distribution distance from training cohort
   - Flags severe visual artifacts, lighting anomalies, or obstruction
3. Human Review Routing:
   - When confidence is below threshold or OOD is detected, outputs:
     "NEEDS_HUMAN_REVIEW"
     with explicit clinical reasoning instead of assigning an uncertain grade.

Pure Python + NumPy ONLY.
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
    """Stack neighbouring window features [x_{t-K}, ..., x_t, ..., x_{t+K}]."""
    T, D = X.shape
    if T == 0:
        return np.zeros((0, (2 * window_radius + 1) * D), dtype=np.float32)

    padded = np.pad(X, ((window_radius, window_radius), (0, 0)), mode="edge")
    context_list = []
    for offset in range(2 * window_radius + 1):
        context_list.append(padded[offset : offset + T])
    return np.hstack(context_list).astype(np.float32)


def load_context_sessions(features_dir: str, gt_path: str, radius: int = 1):
    """Load features with ablation, per-video norm, and context stacking."""
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

        # 163-dim features
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

    X_noise = X.copy()
    X_noise += np.random.normal(0, 0.04, size=(T, D)).astype(np.float32)
    scale = np.random.uniform(0.85, 1.15)
    for k in range(2 * radius + 1):
        offset = k * base_dim
        X_noise[:, offset + 65 : offset + 117] *= scale
    X_aug_list.append(X_noise)
    y_aug_list.append(y)

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
    max_epochs: int = 80,
    patience: int = 15,
    lr: float = 3e-3,
    batch_size: int = 64,
    seed: int = 42,
) -> FoleyStepClassifier:
    """Train MLP with early stopping."""
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


def compute_confidence_metrics(probs: np.ndarray, X_test: np.ndarray, X_train: np.ndarray) -> Dict:
    """Compute multi-level confidence, entropy, and OOD metrics for a session."""
    T = len(probs)
    max_probs = np.max(probs, axis=1)
    mean_conf = float(np.mean(max_probs))
    min_conf = float(np.min(max_probs))
    
    # Ambiguous windows (where max probability is below 0.35)
    amb_mask = (max_probs < 0.35)
    amb_ratio = float(np.mean(amb_mask))
    
    # Normalized Shannon Entropy per window
    # H = -sum(p * log p) / log(14)
    eps = 1e-9
    ent = -np.sum(probs * np.log(probs + eps), axis=1) / np.log(14.0)
    mean_entropy = float(np.mean(ent))
    
    # OOD Check: distance of candidate feature distribution to training distribution
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0) + 1e-6
    test_mean = np.mean(X_test, axis=0)
    
    # Standardized distance from training centroid
    std_dist = np.abs(test_mean - train_mean) / train_std
    ood_score = float(np.mean(std_dist))
    max_ood_dim = float(np.max(std_dist))
    
    is_ood = (ood_score > 1.8) or (max_ood_dim > 5.0)
    
    # Decision: Needs Human Review if:
    # 1. High ambiguous window ratio (> 35%)
    # 2. Mean confidence is very low (< 0.40)
    # 3. High mean entropy (> 0.70)
    # 4. Out-of-distribution feature distance
    needs_review = (amb_ratio > 0.35) or (mean_conf < 0.40) or (mean_entropy > 0.70) or is_ood
    
    reasons = []
    if is_ood:
        reasons.append(f"OOD features (dist={ood_score:.2f})")
    if amb_ratio > 0.35:
        reasons.append(f"{amb_ratio*100:.1f}% ambiguous windows")
    if mean_conf < 0.40:
        reasons.append(f"Low mean confidence ({mean_conf:.2f})")
    if mean_entropy > 0.70:
        reasons.append(f"High predictive entropy ({mean_entropy:.2f})")

    return {
        "mean_confidence": round(mean_conf, 3),
        "min_confidence": round(min_conf, 3),
        "ambiguous_ratio": round(amb_ratio, 3),
        "mean_entropy": round(mean_entropy, 3),
        "ood_score": round(ood_score, 3),
        "is_ood": is_ood,
        "needs_review": needs_review,
        "review_reason": "; ".join(reasons) if reasons else "Confident assessment",
    }


def run_confidence_loso(sessions: Dict[str, Dict], radius: int = 1):
    """Execute 24-fold LOSO with confidence and OOD checks."""
    stems = sorted(list(sessions.keys()))
    n_folds = len(stems)

    all_y_true = []
    all_y_pred_window = []

    session_results = []
    grade_matches = 0
    human_reviews_flagged = 0
    confident_sessions_evaluated = 0

    checker = SessionChecker(min_step_duration_sec=2.0)
    grader = OSCEGrader()

    t_start = time.time()
    print("=" * 115)
    print(f"TASK 6: CONFIDENCE & OUT-OF-DISTRIBUTION (OOD) HUMAN-REVIEW CHECK ({n_folds} FOLDS)")
    print("=" * 115)
    print(f"{'FOLD':5s} | {'HELD-OUT':10s} | {'NAME':14s} | {'WIN ACC':8s} | {'CONF':6s} | {'AMB%':6s} | {'TEACHER':8s} | {'PREDICTED':18s} | {'REVIEW REASON'}")
    print("-" * 115)

    for i, held_out_stem in enumerate(stems):
        train_pool_stems = [s for s in stems if s != held_out_stem]

        val_stems = [train_pool_stems[(i + k) % len(train_pool_stems)] for k in range(4)]
        train_stems = [s for s in train_pool_stems if s not in val_stems]

        X_train_list = []
        y_train_list = []
        for s in train_stems:
            x_aug, y_aug = augment_context_features(sessions[s]["X"], sessions[s]["y"], radius=radius)
            X_train_list.append(x_aug)
            y_train_list.append(y_aug)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        X_val = np.vstack([sessions[s]["X"] for s in val_stems])
        y_val = np.concatenate([sessions[s]["y"] for s in val_stems])

        test_session = sessions[held_out_stem]
        X_test = test_session["X"]
        y_test = test_session["y"]
        gt_info = test_session["gt"]
        cand_name = gt_info.get("candidate_name", held_out_stem)
        faculty_grade = gt_info.get("faculty_grade", "P")
        prompted_steps = gt_info.get("prompted_steps", [])

        model = train_context_model(
            X_train, y_train,
            X_val, y_val,
            max_epochs=150,
            patience=20,
            lr=3e-3,
            seed=i + 400,
        )

        probs, _ = model.forward(X_test)
        raw_preds = np.argmax(probs, axis=1)

        win_acc = float(np.mean(raw_preds == y_test)) * 100.0
        all_y_true.extend(y_test.tolist())
        all_y_pred_window.extend(raw_preds.tolist())

        # Confidence & OOD metrics
        conf_info = compute_confidence_metrics(probs, X_test, X_train)

        # Viterbi decoding & segments
        viterbi_preds = checker.viterbi_decode(probs)
        windows = [{"start_sec": t * 2.0, "end_sec": t * 2.0 + 4.0} for t in range(len(X_test))]
        segments = checker.extract_segments(viterbi_preds, windows, window_probs=probs)

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

        model_raw_grade = eval_report["final_grade"]

        # Confidence Guardrail
        if conf_info["needs_review"]:
            human_reviews_flagged += 1
            final_output = "NEEDS_HUMAN_REVIEW"
            status_str = "[REVIEW]"
        else:
            confident_sessions_evaluated += 1
            final_output = model_raw_grade
            is_match = (model_raw_grade == faculty_grade)
            if is_match:
                grade_matches += 1
            status_str = "[MATCH]" if is_match else "[FAIL]"

        print(f"[{i+1:02d}/{n_folds:02d}] | {held_out_stem:10s} | {cand_name:14s} | {win_acc:6.1f}% | {conf_info['mean_confidence']:4.2f} | {conf_info['ambiguous_ratio']*100:4.1f}% | {faculty_grade:8s} | {final_output:18s} | {conf_info['review_reason']}")

        session_results.append({
            "fold": i + 1,
            "held_out_stem": held_out_stem,
            "candidate_name": cand_name,
            "win_accuracy": win_acc,
            "mean_confidence": conf_info["mean_confidence"],
            "ambiguous_ratio": conf_info["ambiguous_ratio"],
            "mean_entropy": conf_info["mean_entropy"],
            "ood_score": conf_info["ood_score"],
            "teacher_grade": faculty_grade,
            "model_raw_grade": model_raw_grade,
            "final_output": final_output,
            "needs_human_review": conf_info["needs_review"],
            "review_reason": conf_info["review_reason"],
            "status": status_str,
        })

    elapsed = time.time() - t_start

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred_window)
    overall_win_acc = float(np.mean(y_true_arr == y_pred_arr)) * 100.0

    print("\n" + "=" * 95)
    print(f"TASK 6 CONFIDENCE & OOD GUARDRAIL SUMMARY ({elapsed:.1f}s total)")
    print("=" * 95)
    print(f"Total Held-Out Candidates:          {n_folds}")
    print(f"Sessions Flagged for Human Review:  {human_reviews_flagged}/{n_folds} ({human_reviews_flagged/n_folds*100:.1f}%)")
    print(f"Sessions Evaluated Confidently:     {confident_sessions_evaluated}/{n_folds} ({confident_sessions_evaluated/n_folds*100:.1f}%)")
    if confident_sessions_evaluated > 0:
        conf_acc = (grade_matches / confident_sessions_evaluated) * 100.0
        print(f"Accuracy on Confident Sessions:     {grade_matches}/{confident_sessions_evaluated} ({conf_acc:.1f}%)")
    print(f"Overall Window Accuracy:            {overall_win_acc:.2f}%")
    print("-" * 95)

    out_json = {
        "n_folds": n_folds,
        "elapsed_seconds": round(elapsed, 1),
        "overall_window_accuracy": round(overall_win_acc, 2),
        "human_reviews_flagged": human_reviews_flagged,
        "confident_sessions_evaluated": confident_sessions_evaluated,
        "confident_grade_matches": grade_matches,
        "session_results": session_results,
    }

    report_path = r"C:\Medical\reports\loso_confidence_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved detailed LOSO confidence report to: {report_path}")

    return out_json


if __name__ == "__main__":
    sessions = load_context_sessions(
        features_dir=r"C:\Medical\dataset\features_128px",
        gt_path=r"C:\Medical\dataset\cohort_ground_truth.json",
        radius=1,
    )
    run_confidence_loso(sessions, radius=1)
