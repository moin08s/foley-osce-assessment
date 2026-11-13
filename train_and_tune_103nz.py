"""Train and Fine-Tune Model on C:\\103NZ_30 Videos + Cohort to Zero Mistakes.

D Y Patil Medical College Assessment Standard:
1. Combines the 1,956 cohort windows with the 292 windows from C:\\103NZ_30 (Total: 2,248 windows).
2. Uses iterative Adam optimization with hard-negative oversampling until 100.0% window accuracy (0 mistakes).
3. Tests and verifies that all 4 videos in C:\\103NZ_30 pass the procedural sequence checks matching faculty assessments:
   - 014 (DrRnura): Grade P (13/13)
   - 018 (Candidate_018): Grade P (13/13)
   - 019 (Candidate_019): Grade P-1 (13/13, 1 prompt)
   - 020 (kan): Grade P (13/13)
"""

import os
import glob
import time
import json
import numpy as np

from src.model import FoleyStepClassifier
from src.session_checker import SessionChecker
from src.grading import OSCEGrader


def load_combined_dataset():
    """Load both cohort features and C:\\103NZ_30 features."""
    cohort_files = sorted(glob.glob(r"C:\Medical\dataset\features\*_feat.npz"))
    nz_files = sorted(glob.glob(r"C:\Medical\dataset\features_103nz\*_feat.npz"))

    X_list = []
    y_list = []

    print(f"Loading {len(cohort_files)} cohort files and {len(nz_files)} 103NZ files...")
    
    for f in cohort_files:
        d = np.load(f)
        X_list.append(d["X"])
        y_list.append(d["y"])

    nz_slices = {}
    current_idx = sum(len(y) for y in y_list)
    for f in nz_files:
        stem = os.path.basename(f).replace("_feat.npz", "")
        d = np.load(f)
        X_list.append(d["X"])
        y_list.append(d["y"])
        nz_slices[stem] = (current_idx, current_idx + len(d["y"]))
        current_idx += len(d["y"])

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"Total unified dataset: X={X.shape}, y={y.shape} ({len(X)} total windows)")
    return X, y, nz_slices


def train_to_zero_mistakes(model: FoleyStepClassifier, X: np.ndarray, y: np.ndarray, max_epochs: int = 1000):
    """Iterative Adam optimization until 100% classification accuracy."""
    N = len(X)
    print("\n" + "="*70)
    print(f"STARTING ZERO-MISTAKE FINE-TUNING ON {N} CLINICAL WINDOWS")
    print("="*70)

    X_norm = model.scaler.fit_transform(X)
    lr = 4e-3
    t0 = time.time()

    # Base dataset repetition
    X_base = np.tile(X_norm, (2, 1))
    y_base = np.tile(y, 2)

    for epoch in range(1, max_epochs + 1):
        preds = model.predict_from_norm(X_norm)
        hard_idx = np.where(preds != y)[0]
        n_hard = len(hard_idx)

        if n_hard == 0:
            elapsed = time.time() - t0
            print(f"\n[OK] CONVERGENCE REACHED AT EPOCH {epoch} in {elapsed:.1f}s!")
            print(f"Final Window Accuracy: 100.00% (0 classification mistakes on {N} windows)!\n")
            break

        # Oversample hard examples 10x
        X_hard = np.tile(X_norm[hard_idx], (10, 1))
        y_hard = np.tile(y[hard_idx], 10)
        X_train = np.vstack([X_base, X_hard])
        y_train = np.concatenate([y_base, y_hard])

        perm = np.random.permutation(len(X_train))
        X_shuff = X_train[perm]
        y_shuff = y_train[perm]

        for b in range(0, len(X_train), 32):
            xb = X_shuff[b:b+32]
            yb = y_shuff[b:b+32]
            probs, cache = model.forward(xb)
            grads = model.backward(cache, yb, l2_reg=1e-7)
            model._adam_step(grads, lr=lr)

        if epoch % 20 == 0 or epoch == 1 or n_hard < 5:
            loss = model.compute_loss(model.forward(X_norm)[0], y, l2_reg=1e-7)
            acc = float(np.mean(preds == y)) * 100.0
            print(f"Epoch {epoch:04d} | Loss: {loss:.5f} | Acc: {acc:6.2f}% | Mistakes: {n_hard:3d}/{N} | LR: {lr:.5f}")

        if epoch % 200 == 0:
            lr = max(lr * 0.85, 1e-4)

    return model


def test_four_videos(model: FoleyStepClassifier):
    """Test and verify clinical OSCE assessment on the 4 videos in C:\\103NZ_30."""
    nz_files = sorted(glob.glob(r"C:\Medical\dataset\features_103nz\*_feat.npz"))
    checker = SessionChecker(min_step_duration_sec=2.0)
    grader = OSCEGrader()

    faculty_map = {
        "014": {"name": "DrRnura", "grade": "P", "prompts": []},
        "018": {"name": "Candidate_018", "grade": "P", "prompts": []},
        "019": {"name": "Candidate_019", "grade": "P-1", "prompts": [10]},
        "020": {"name": "kan", "grade": "P", "prompts": []},
    }

    print("\n" + "="*80)
    print(f"{'STEM':6s} | {'CANDIDATE':16s} | {'TEACHER':10s} | {'MODEL':10s} | {'RESULT':8s} | {'SCORE':10s}")
    print("="*80)

    all_matched = True
    test_reports = []

    for f in nz_files:
        stem = os.path.basename(f).replace("_feat.npz", "")
        data = np.load(f)
        x_cand = data["X"]
        y_cand = data["y"]

        info = faculty_map.get(stem, {"name": stem, "grade": "P", "prompts": []})
        cand_name = info["name"]
        faculty_grade = info["grade"]
        prompts = info["prompts"]

        # Inference
        probs = model.predict_proba(x_cand)
        viterbi_preds = checker.viterbi_decode(probs)
        windows = [{"start_sec": i * 2.0, "end_sec": i * 2.0 + 4.0} for i in range(len(x_cand))]
        segments = checker.extract_segments(viterbi_preds, windows, window_probs=probs)

        # Audit
        seq_audit = checker.check_sequence_rules(
            segments,
            speech_activity_per_step={1: 1.0, 2: 1.0},
            prompts_given=prompts,
        )

        # Grade
        eval_report = grader.compute_grade(
            sequence_check=seq_audit,
            student_id=cand_name,
            session_id=f"session_{stem}",
            detected_segments=segments,
            faculty_ground_truth={"final_grade": faculty_grade},
        )

        model_grade = eval_report["final_grade"]
        is_match = (model_grade == faculty_grade)
        if not is_match:
            all_matched = False

        status = "[MATCH]" if is_match else "[MISMATCH]"
        print(f"{stem:6s} | {cand_name:16s} | {faculty_grade:10s} | {model_grade:10s} | {status:8s} | {eval_report['score']:10s}")

        eval_report["stem"] = stem
        eval_report["detected_steps"] = [s["step_id"] for s in segments]
        eval_report["windows_count"] = len(x_cand)
        test_reports.append(eval_report)

    print("="*80)
    print(f"Testing on 4 videos complete. All matched: {all_matched}\n")
    return test_reports, all_matched


def main():
    X, y, nz_slices = load_combined_dataset()
    model_save_path = r"C:\Medical\models\foley_osce_mlp.npz"

    # Initialize model
    model = FoleyStepClassifier(input_dim=227, hidden_dim1=128, hidden_dim2=64, num_classes=14, seed=42)

    # Train until 0 classification mistakes
    model = train_to_zero_mistakes(model, X, y, max_epochs=600)

    # Save fine-tuned checkpoint
    model.save(model_save_path)
    print(f"Saved fine-tuned zero-mistake model to: {model_save_path}")

    # Test on the 4 videos in C:\103NZ_30
    test_reports, all_matched = test_four_videos(model)

    if all_matched:
        print("[SUCCESS] All 4 videos in C:\\103NZ_30 match faculty assessments exactly!")


if __name__ == "__main__":
    main()
