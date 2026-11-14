"""Leave-One-Candidate-Out (LOSO) Cross-Validation for Male Foley OSCE Assessment.

Task 1:
- 24 Folds: Train on 23 candidates, evaluate strictly on the 1 held-out candidate.
- Never fit scaler or train on the test candidate (zero data leakage).
- Evaluates:
  1. Window-level classification accuracy.
  2. Per-step Precision, Recall, and F1 score (steps 1 to 13, and class 0).
  3. Segment-level step detection rate (true step present vs detected).
  4. Final OSCE Grade Agreement (P / P-1 / R) against teacher ground truth.
- Pure Python Standard Library + NumPy only.
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


def load_all_sessions(features_dir: str, gt_path: str):
    """Load features and metadata for all 24 candidates."""
    with open(gt_path, "r", encoding="utf-8") as f:
        cohort_gt = json.load(f)

    session_files = sorted(glob.glob(os.path.join(features_dir, "*_feat.npz")))
    if not session_files:
        raise FileNotFoundError(f"No *_feat.npz found in {features_dir}")

    sessions = {}
    for f in session_files:
        stem = os.path.basename(f).replace("_feat.npz", "")
        if stem not in cohort_gt:
            continue
        data = np.load(f)
        sessions[stem] = {
            "stem": stem,
            "X": data["X"].astype(np.float32),
            "y": data["y"].astype(np.int64),
            "gt": cohort_gt[stem],
        }

    print(f"Loaded {len(sessions)} valid candidate sessions for LOSO validation.")
    return sessions


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 160,
    lr: float = 3e-3,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[FoleyStepClassifier, StandardScaler]:
    """Train classifier on training sessions only."""
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)

    model = FoleyStepClassifier(
        input_dim=X_train.shape[1],
        hidden_dim1=128,
        hidden_dim2=64,
        num_classes=14,
        seed=seed,
    )
    # Assign scaler parameters so model.predict_proba works if needed
    model.scaler.mean = scaler.mean
    model.scaler.scale = scaler.scale

    N = len(X_train)
    curr_lr = lr

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N)
        X_shuff = X_train_norm[perm]
        y_shuff = y_train[perm]

        for b_start in range(0, N, batch_size):
            b_end = min(b_start + batch_size, N)
            xb = X_shuff[b_start:b_end]
            yb = y_shuff[b_start:b_end]
            probs, cache = model.forward(xb)
            grads = model.backward(cache, yb, l2_reg=1e-5)
            model._adam_step(grads, lr=curr_lr)

        if epoch % 80 == 0:
            curr_lr *= 0.7

    return model, scaler


def run_loso_cross_validation(
    sessions: Dict[str, Dict],
    epochs_per_fold: int = 160,
) -> Dict[str, any]:
    """Execute complete 24-fold LOSO cross-validation."""
    stems = sorted(list(sessions.keys()))
    n_folds = len(stems)

    all_y_true = []
    all_y_pred_window = []

    # Segment detection tracking: step_id -> {present: count, detected: count}
    step_detection_counts = {s: {"present": 0, "detected": 0} for s in range(1, 14)}

    session_results = []
    grade_matches = 0
    grade_confusion = {"P": {"P": 0, "P-1": 0, "R": 0}, "P-1": {"P": 0, "P-1": 0, "R": 0}, "R": {"P": 0, "P-1": 0, "R": 0}}

    checker = SessionChecker(min_step_duration_sec=2.0)
    grader = OSCEGrader()

    t_start = time.time()
    print("=" * 95)
    print(f"STARTING LEAVE-ONE-CANDIDATE-OUT (LOSO) CROSS-VALIDATION ({n_folds} FOLDS)")
    print("Strict Protocol: Zero data leakage. Scaler and model fit only on 23 training sessions.")
    print("=" * 95)
    print(f"{'FOLD':5s} | {'HELD-OUT':10s} | {'NAME':14s} | {'WIN ACC':8s} | {'TEACHER':8s} | {'MODEL':8s} | {'RESULT':8s} | {'SCORE':8s}")
    print("-" * 95)

    for i, held_out_stem in enumerate(stems):
        train_stems = [s for s in stems if s != held_out_stem]

        # 1. Build train set from 23 other candidates
        X_train_list = [sessions[s]["X"] for s in train_stems]
        y_train_list = [sessions[s]["y"] for s in train_stems]
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # 2. Held-out test candidate
        test_session = sessions[held_out_stem]
        X_test = test_session["X"]
        y_test = test_session["y"]
        gt_info = test_session["gt"]
        cand_name = gt_info.get("candidate_name", held_out_stem)
        faculty_grade = gt_info.get("faculty_grade", "P")
        prompted_steps = gt_info.get("prompted_steps", [])

        # 3. Train on 23 sessions
        model, scaler = train_fold(X_train, y_train, epochs=epochs_per_fold, lr=3e-3, seed=i + 42)

        # 4. Predict on held-out test candidate
        X_test_norm = scaler.transform(X_test)
        probs, _ = model.forward(X_test_norm)
        raw_preds = np.argmax(probs, axis=1)

        win_acc = float(np.mean(raw_preds == y_test)) * 100.0
        all_y_true.extend(y_test.tolist())
        all_y_pred_window.extend(raw_preds.tolist())

        # 5. Temporal decoding and segment extraction
        viterbi_preds = checker.viterbi_decode(probs)
        windows = [{"start_sec": t * 2.0, "end_sec": t * 2.0 + 4.0} for t in range(len(X_test))]
        segments = checker.extract_segments(viterbi_preds, windows, window_probs=probs)
        detected_step_ids = set(s["step_id"] for s in segments)

        # Segment-level step detection tracking
        true_steps_present = set(y_test.tolist()) - {0}
        for s in range(1, 14):
            if s in true_steps_present:
                step_detection_counts[s]["present"] += 1
                if s in detected_step_ids:
                    step_detection_counts[s]["detected"] += 1

        # 6. Sequence checking and grading
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
            "missing_steps": seq_audit.get("missing_steps", []),
        })

    elapsed = time.time() - t_start

    # Global window metrics
    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred_window)
    overall_win_acc = float(np.mean(y_true_arr == y_pred_arr)) * 100.0

    # Per-step precision, recall, f1
    per_step_metrics = {}
    for c in range(14):
        tp = int(np.sum((y_true_arr == c) & (y_pred_arr == c)))
        fp = int(np.sum((y_true_arr != c) & (y_pred_arr == c)))
        fn = int(np.sum((y_true_arr == c) & (y_pred_arr != c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(np.sum(y_true_arr == c))

        per_step_metrics[c] = {
            "name": STEP_LABELS.get(c, f"Step {c}"),
            "precision": prec * 100.0,
            "recall": rec * 100.0,
            "f1": f1 * 100.0,
            "support": support,
        }

    grade_agreement_pct = (grade_matches / n_folds) * 100.0

    print("\n" + "=" * 95)
    print(f"LOSO EVALUATION SUMMARY ({elapsed:.1f}s total)")
    print("=" * 95)
    print(f"Total Held-Out Candidates:       {n_folds}")
    print(f"Held-Out Final Grade Agreement: {grade_matches}/{n_folds} ({grade_agreement_pct:.1f}%)")
    print(f"Held-Out Overall Window Accuracy: {overall_win_acc:.2f}%")
    print("-" * 95)

    print("\nPER-STEP HELD-OUT CLASSIFICATION METRICS:")
    print(f"{'CLASS':5s} | {'STEP NAME':30s} | {'PRECISION':10s} | {'RECALL':10s} | {'F1-SCORE':10s} | {'SUPPORT':8s}")
    print("-" * 95)
    for c in range(14):
        m = per_step_metrics[c]
        print(f"{c:5d} | {m['name']:30s} | {m['precision']:9.1f}% | {m['recall']:9.1f}% | {m['f1']:9.1f}% | {m['support']:8d}")

    print("\nSEGMENT-LEVEL STEP DETECTION RATE (SENSITIVITY ON HELD-OUT SESSIONS):")
    print(f"{'STEP':5s} | {'STEP NAME':30s} | {'PRESENT':8s} | {'DETECTED':8s} | {'DETECTION RATE':14s}")
    print("-" * 95)
    for s in range(1, 14):
        cnt = step_detection_counts[s]
        pres = cnt["present"]
        det = cnt["detected"]
        rate = (det / pres * 100.0) if pres > 0 else 0.0
        print(f"{s:5d} | {STEP_LABELS[s]:30s} | {pres:8d} | {det:8d} | {rate:13.1f}%")

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
        "per_step_metrics": per_step_metrics,
        "step_detection_counts": step_detection_counts,
        "grade_confusion": grade_confusion,
        "session_results": session_results,
    }

    report_path = r"C:\Medical\reports\loso_baseline_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved detailed LOSO baseline report to: {report_path}")

    return out_json


if __name__ == "__main__":
    sessions = load_all_sessions(
        features_dir=r"C:\Medical\dataset\features",
        gt_path=r"C:\Medical\dataset\cohort_ground_truth.json",
    )
    run_loso_cross_validation(sessions, epochs_per_fold=160)
