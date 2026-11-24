"""Phase 5: Full 23-Fold LOSO Evaluation with De-Biased 123-D Features & Viterbi Decoder.

D Y Patil Medical College - OSCE Procedural Skills Assessment Model
Strict Leave-One-Candidate-Out (LOSO) Cross-Validation across all 23 valid candidates.
Zero data leakage: Every metric is evaluated strictly on an unseen, held-out student.

Compares:
1. Phase 4 Baseline MLP (18.31%)
2. Phase 4 Best Config E (29.07%)
3. Phase 2 Time-Position Baseline (68.90% window acc, 63.44% seg acc, 56.52% grade agr)
4. Phase 5 De-Biased 123-D + Viterbi (Target: Beat the baseline floor)
"""

import os
import glob
import json
import time
import numpy as np
from typing import Dict, List, Tuple

BASE_DIR = r"C:\Medical"
GT_PATH = os.path.join(BASE_DIR, "dataset", "cohort_ground_truth.json")
FEAT_DIR = os.path.join(BASE_DIR, "dataset", "features_clean_123")
OUT_REPORT = os.path.join(BASE_DIR, "reports", "phase5_overcome_bottlenecks_report.json")

from train_phase4_loso_model import PureNumpyMLP
from viterbi_decoder import viterbi_procedural_decode
from src.session_checker import SessionChecker
from src.grading import OSCEGrader


def load_clean_dataset() -> Dict[str, dict]:
    with open(GT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, "*_feat.npz")))
    sessions = {}
    for f in feat_files:
        stem = os.path.basename(f).replace("_feat.npz", "")
        if stem not in cohort or stem == "025":
            continue

        data = np.load(f)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(int)
        T = len(y)

        # Apply per-video z-score standardization on motion (0:65) and audio (69:115)
        # Leave clinical colors (65:69) and temporal harmonics (115:123) unstandardized
        motion = X[:, 0:65]
        m_mean = np.mean(motion, axis=0, keepdims=True)
        m_std = np.std(motion, axis=0, keepdims=True) + 1e-6
        motion_norm = (motion - m_mean) / m_std

        colors = X[:, 65:69]

        audio = X[:, 69:115]
        a_mean = np.mean(audio, axis=0, keepdims=True)
        a_std = np.std(audio, axis=0, keepdims=True) + 1e-6
        audio_norm = (audio - a_mean) / a_std

        temporal = X[:, 115:123]

        X_norm = np.hstack([motion_norm, colors, audio_norm, temporal]).astype(np.float32)

        sessions[stem] = {
            "stem": stem,
            "name": cohort[stem].get("student_name", f"Candidate_{stem}"),
            "faculty_grade": cohort[stem]["faculty_grade"],
            "X": X_norm,
            "y": y,
            "T": T,
        }
    return sessions


def run_phase5_loso_benchmark():
    sessions = load_clean_dataset()
    stems = sorted(list(sessions.keys()))
    print("=" * 110)
    print("PHASE 5: FULL 23-FOLD LEAVE-ONE-CANDIDATE-OUT (LOSO) BENCHMARK")
    print(f"Total Evaluated Candidates: {len(stems)} (Excluding 025 and 006)")
    print("Features: De-biased 123-D (Identity-Confounding Dropped + 8-D Fourier Progression Harmonics)")
    print("Decoder: Duration-Constrained Monotonic Viterbi Decoder")
    print("=" * 110)

    session_checker = SessionChecker()
    grader = OSCEGrader()

    candidate_results = []
    total_windows_correct_raw = 0
    total_windows_correct_vit = 0
    total_windows_all = 0
    total_segment_hits = 0
    total_segments_possible = 0
    grade_matches = 0

    t_start = time.time()

    print(f"{'STEM':4s} | {'STUDENT':14s} | {'GRADE':5s} | {'PRED':5s} | {'WIN ACC (RAW)':14s} | {'WIN ACC (VIT)':14s} | {'SEG ACC (±4s)':14s} | {'MATCH?'}")
    print("-" * 110)

    for test_stem in stems:
        train_stems = [s for s in stems if s != test_stem]

        # 1. Prepare Training Fold
        X_train_list = [sessions[s]["X"] for s in train_stems]
        y_train_list = [sessions[s]["y"] for s in train_stems]
        X_tr = np.vstack(X_train_list)
        y_tr = np.concatenate(y_train_list)

        X_te = sessions[test_stem]["X"]
        y_te = sessions[test_stem]["y"]
        T_te = len(y_te)

        # 2. Compute Class Weights for Imbalance
        unique_classes, counts = np.unique(y_tr, return_counts=True)
        class_weights = np.ones(14, dtype=np.float32)
        total_samples = len(y_tr)
        for c, count in zip(unique_classes, counts):
            class_weights[c] = total_samples / (14.0 * max(1, count))
        # Normalize weights
        class_weights /= np.mean(class_weights)

        # 3. Train Pure NumPy MLP
        in_dim = X_tr.shape[1]
        model = PureNumpyMLP(in_dim=in_dim, h1=128, h2=64, num_classes=14, seed=42)
        batch_size = 64
        n_samples = len(y_tr)
        lr = 0.003

        for epoch in range(85):
            perm = np.random.RandomState(epoch).permutation(n_samples)
            for b in range(0, n_samples, batch_size):
                idx = perm[b:b+batch_size]
                probs, cache = model.forward(X_tr[idx])
                # Compute gradients
                grads = model.backward(cache, y_tr[idx], l2_reg=1e-3)
                model.adam_step(grads, lr=lr)

        # 4. Predict on Unseen Candidate
        probs_te, _ = model.forward(X_te)
        y_pred_raw = np.argmax(probs_te, axis=-1)
        raw_corr = int(np.sum(y_pred_raw == y_te))

        # 5. Viterbi Monotonic Decode
        y_pred_vit = viterbi_procedural_decode(probs_te, min_duration_windows=2)
        vit_corr = int(np.sum(y_pred_vit == y_te))

        total_windows_correct_raw += raw_corr
        total_windows_correct_vit += vit_corr
        total_windows_all += T_te

        raw_acc = (raw_corr / T_te) * 100.0
        vit_acc = (vit_corr / T_te) * 100.0

        # 6. Segment Boundary Detection (±4s tolerance)
        mock_windows = [{"start_sec": i * 2.0, "end_sec": i * 2.0 + 4.0} for i in range(T_te)]
        pred_segments = session_checker.extract_segments(y_pred_vit, mock_windows, probs_te)
        true_segments = session_checker.extract_segments(y_te, mock_windows)

        hits = 0
        for ts in true_segments:
            sid = ts["step_id"]
            matched = False
            for ps in pred_segments:
                if ps["step_id"] == sid:
                    if abs(ps["start_sec"] - ts["start_sec"]) <= 4.0:
                        matched = True
                        break
            if matched:
                hits += 1

        total_segment_hits += hits
        total_segments_possible += len(true_segments)
        seg_acc = (hits / max(1, len(true_segments))) * 100.0

        # 7. Sequence Validation & Grade Assignment
        seq_check = session_checker.check_sequence_rules(pred_segments)
        grade_res = grader.compute_grade(
            sequence_check=seq_check,
            student_id=sessions[test_stem]["name"],
            session_id=test_stem,
            detected_segments=pred_segments,
        )
        pred_grade = grade_res["final_grade"]
        true_grade = sessions[test_stem]["faculty_grade"]
        match = (pred_grade == true_grade)
        if match:
            grade_matches += 1

        print(f"{test_stem:4s} | {sessions[test_stem]['name'][:14]:14s} | {true_grade:5s} | {pred_grade:5s} | {raw_acc:6.2f}% ({raw_corr}/{T_te}) | {vit_acc:6.2f}% ({vit_corr}/{T_te}) | {seg_acc:6.2f}% ({hits}/{len(true_segments)}) | {'MATCH' if match else 'DIFF'}")

        candidate_results.append({
            "stem": test_stem,
            "name": sessions[test_stem]["name"],
            "true_grade": true_grade,
            "pred_grade": pred_grade,
            "grade_match": match,
            "windows": T_te,
            "raw_acc": raw_acc,
            "vit_acc": vit_acc,
            "segment_hits": hits,
            "segment_total": len(true_segments),
            "segment_acc": seg_acc,
        })

    elapsed = time.time() - t_start
    overall_raw_acc = (total_windows_correct_raw / total_windows_all) * 100.0
    overall_vit_acc = (total_windows_correct_vit / total_windows_all) * 100.0
    overall_seg_acc = (total_segment_hits / total_segments_possible) * 100.0
    overall_grade_agr = (grade_matches / len(stems)) * 100.0

    print("=" * 110)
    print("PHASE 5 LOSO SUMMARY COMPARISON:")
    print(f"  Overall Raw Window Accuracy (Argmax):     {overall_raw_acc:.2f}% ({total_windows_correct_raw}/{total_windows_all})")
    print(f"  Overall Viterbi Window Accuracy:          {overall_vit_acc:.2f}% ({total_windows_correct_vit}/{total_windows_all})")
    print(f"  Overall Segment Detection Accuracy (±4s): {overall_seg_acc:.2f}% ({total_segment_hits}/{total_segments_possible})")
    print(f"  Grade Agreement with Faculty:             {overall_grade_agr:.2f}% ({grade_matches}/{len(stems)})")
    print(f"  Total 23-Fold Cross-Validation Runtime:   {elapsed:.1f} s")
    print("=" * 110)

    # Comparison benchmarks
    benchmarks = {
        "Phase 4 Baseline MLP (227-D raw)": {"window_acc": 18.31, "segment_acc": 30.47, "grade_agr": 21.74},
        "Phase 4 Config E (681-D context + Viterbi)": {"window_acc": 29.07, "segment_acc": 25.45, "grade_agr": 21.74},
        "Phase 2 Time-Position Baseline (Prior Floor)": {"window_acc": 68.90, "segment_acc": 63.44, "grade_agr": 56.52},
        "Phase 5 De-Biased 123-D + Viterbi (Ours)": {"window_acc": round(overall_vit_acc, 2), "segment_acc": round(overall_seg_acc, 2), "grade_agr": round(overall_grade_agr, 2)},
    }

    report = {
        "evaluation_protocol": "Strict Leave-One-Candidate-Out (LOSO) 23-Fold Cross-Validation",
        "features": "De-biased 123-D (Motion + Procedural Colors + Audio + Temporal Harmonic Progression)",
        "decoder": "Duration-Constrained Monotonic Viterbi",
        "cohort_size": len(stems),
        "metrics": {
            "overall_raw_window_accuracy": round(overall_raw_acc, 2),
            "overall_viterbi_window_accuracy": round(overall_vit_acc, 2),
            "overall_segment_accuracy_4s": round(overall_seg_acc, 2),
            "faculty_grade_agreement_pct": round(overall_grade_agr, 2),
            "grade_matches": grade_matches,
            "total_candidates": len(stems),
            "runtime_seconds": round(elapsed, 1),
        },
        "benchmarks_comparison": benchmarks,
        "candidate_results": candidate_results,
    }

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved full Phase 5 report to: {OUT_REPORT}")
    return report

if __name__ == "__main__":
    run_phase5_loso_benchmark()
