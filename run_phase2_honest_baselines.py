"""Phase 2: Honest Baselines Evaluation (Strict Leave-One-Candidate-Out).

Using ONLY human_annotated rows:
1. Fits time-position baseline proportions strictly on training folds (no data leakage).
2. Evaluates window accuracy, segment accuracy (+/- 4s tolerance) on held-out candidate.
3. Computes majority-grade baseline on held-out candidates.
4. Reports per-step done/missed accuracy alongside the number of negative sessions.
"""

import os
import json
import csv
import numpy as np
from typing import Dict, List, Tuple

BASE_DIR = r"C:\Medical"
GT_PATH = os.path.join(BASE_DIR, "dataset", "cohort_ground_truth.json")
ANN_PATH = os.path.join(BASE_DIR, "human_annotation_sheet.csv")
FEAT_DIR = os.path.join(BASE_DIR, "dataset", "features_128px")
OUT_REPORT = os.path.join(BASE_DIR, "reports", "phase2_honest_baselines_report.json")

from src.data_loader import STEP_LABELS

def load_human_annotations() -> Dict[str, dict]:
    with open(ANN_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    sessions = {}
    for r in rows:
        v = r["video"].replace(".MP4", "").replace(".mp4", "")
        sid = int(r["step_id"])
        if v not in sessions:
            sessions[v] = {
                "stem": v,
                "steps": {},
                "annotator": r.get("annotator_name", ""),
            }
        
        status = r.get("status", "").lower()
        start_str = r.get("start_sec", "").strip()
        end_str = r.get("end_sec", "").strip()

        start_val = float(start_str) if start_str else None
        end_val = float(end_str) if end_str else None

        sessions[v]["steps"][sid] = {
            "status": status,
            "start_sec": start_val,
            "end_sec": end_val,
            "prompt": r.get("prompt_time_sec", ""),
            "notes": r.get("notes", ""),
        }
    return sessions

def run_phase2():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    sessions = load_human_annotations()
    stems = sorted([s for s in sessions.keys() if s in cohort])

    print("=" * 105)
    print("PHASE 2: HONEST BASELINES REPORT (LEAVE-ONE-CANDIDATE-OUT CROSS-VALIDATION)")
    print(f"Total Evaluated Sessions: {len(stems)} (Using ONLY human_annotated ground truth)")
    print("=" * 105)

    # 1. Per-Step Done / Missed Counts and Accuracies
    step_negatives = {s: [] for s in range(1, 14)}
    step_positives = {s: [] for s in range(1, 14)}

    for s in stems:
        for sid in range(1, 14):
            info = sessions[s]["steps"][sid]
            if info["status"] == "missed" or info["start_sec"] is None:
                step_negatives[sid].append(s)
            else:
                step_positives[sid].append(s)

    print("\n--- 1. PER-STEP DONE / MISSED COHORT DISTRIBUTION ---")
    print(f"{'STEP':<4} | {'STEP LABEL':<30} | {'DONE (POS)':<12} | {'MISSED (NEG)':<14} | {'MAJORITY ACCURACY'}")
    print("-" * 80)
    for sid in range(1, 14):
        n_pos = len(step_positives[sid])
        n_neg = len(step_negatives[sid])
        maj_acc = max(n_pos, n_neg) / len(stems) * 100.0
        neg_str = f"{n_neg} ({', '.join(step_negatives[sid]) if n_neg > 0 else 'None'})"
        print(f"{sid:<4} | {STEP_LABELS[sid]:<30} | {n_pos:<12} | {neg_str:<14} | {maj_acc:5.1f}%")

    # 2. Strict LOSO Time-Position Baseline & Majority Grade Baseline
    all_window_true = []
    all_window_pred = []
    segment_hits = []
    segment_totals = []
    grade_true_list = []
    grade_pred_majority = []

    per_candidate_results = []

    for test_stem in stems:
        train_stems = [s for s in stems if s != test_stem]

        # A. Fit step duration proportions strictly on training folds
        step_dur_sums = {sid: 0.0 for sid in range(1, 14)}
        total_train_dur = 0.0

        for tr in train_stems:
            for sid in range(1, 14):
                info = sessions[tr]["steps"][sid]
                if info["start_sec"] is not None and info["end_sec"] is not None:
                    d = max(0.0, info["end_sec"] - info["start_sec"])
                    step_dur_sums[sid] += d
                    total_train_dur += d

        # Normalized empirical proportions for training fold
        train_proportions = np.array([step_dur_sums[sid] / total_train_dur for sid in range(1, 14)])
        cum_proportions = np.cumsum(train_proportions)

        # B. Majority Grade in training folds
        train_grades = [cohort[tr]["faculty_grade"] for tr in train_stems]
        unique_g, counts_g = np.unique(train_grades, return_counts=True)
        majority_grade = unique_g[np.argmax(counts_g)]

        # C. Evaluate on test candidate
        test_info = sessions[test_stem]
        cand_grade_true = cohort[test_stem]["faculty_grade"]
        grade_true_list.append(cand_grade_true)
        grade_pred_majority.append(majority_grade)

        # Build ground-truth time series for test candidate
        # Determine total duration
        max_end = 0.0
        for sid in range(1, 14):
            e = test_info["steps"][sid]["end_sec"]
            if e is not None and e > max_end:
                max_end = e
        if max_end == 0.0:
            max_end = 150.0

        # Discretize into 2-second windows (hop=2.0s, window=4.0s)
        n_windows = max(1, int(np.ceil((max_end - 4.0) / 2.0)) + 1)
        
        y_true_cand = np.zeros(n_windows, dtype=int)
        y_pred_baseline = np.zeros(n_windows, dtype=int)

        for w_idx in range(n_windows):
            w_mid = w_idx * 2.0 + 2.0
            
            # True label
            true_step = 0
            for sid in range(1, 14):
                st = test_info["steps"][sid]["start_sec"]
                en = test_info["steps"][sid]["end_sec"]
                if st is not None and en is not None and (st <= w_mid < en):
                    true_step = sid
                    break
            y_true_cand[w_idx] = true_step

            # Fitted time-position baseline prediction
            norm_pos = min(0.9999, w_mid / max_end)
            pred_step = int(np.searchsorted(cum_proportions, norm_pos) + 1)
            y_pred_baseline[w_idx] = min(13, max(1, pred_step))

        all_window_true.extend(y_true_cand)
        all_window_pred.extend(y_pred_baseline)

        cand_win_acc = np.mean(y_true_cand == y_pred_baseline) * 100.0

        # Segment level detection (+/- 4s tolerance on step starts)
        cand_hits = 0
        cand_total = 0
        for sid in range(1, 14):
            real_start = test_info["steps"][sid]["start_sec"]
            if real_start is not None:
                cand_total += 1
                # Baseline predicted start time
                prop_start = (cum_proportions[sid-2] if sid > 1 else 0.0) * max_end
                if abs(prop_start - real_start) <= 4.0:
                    cand_hits += 1

        segment_hits.append(cand_hits)
        segment_totals.append(cand_total)

        per_candidate_results.append({
            "stem": test_stem,
            "name": cohort[test_stem]["candidate_name"],
            "true_grade": cand_grade_true,
            "pred_majority_grade": majority_grade,
            "grade_match": cand_grade_true == majority_grade,
            "windows": n_windows,
            "window_acc": cand_win_acc,
            "segment_hits": cand_hits,
            "segment_total": cand_total,
            "segment_acc": (cand_hits / cand_total * 100.0) if cand_total > 0 else 0.0
        })

    # Summary Metrics
    y_true_all = np.array(all_window_true)
    y_pred_all = np.array(all_window_pred)

    overall_window_acc = np.mean(y_true_all == y_pred_all) * 100.0
    overall_segment_acc = np.sum(segment_hits) / np.sum(segment_totals) * 100.0
    overall_grade_acc = np.mean(np.array(grade_true_list) == np.array(grade_pred_majority)) * 100.0
    grade_matches = np.sum(np.array(grade_true_list) == np.array(grade_pred_majority))

    print("\n--- 2. STRICT LEAVE-ONE-CANDIDATE-OUT (LOSO) BASELINE RESULTS ---")
    print(f"1. Window Accuracy (Time-Position Baseline):  {overall_window_acc:5.2f}%")
    print(f"2. Segment Detection (+/- 4s tolerance):      {overall_segment_acc:5.2f}% ({np.sum(segment_hits)}/{np.sum(segment_totals)} steps)")
    print(f"3. Majority-Grade Baseline Agreement:         {overall_grade_acc:5.2f}% ({grade_matches}/{len(stems)} candidates)")
    print("-" * 80)

    # Candidate Level Breakdown Table
    print("\n--- 3. CANDIDATE-LEVEL HELD-OUT BASELINE PERFORMANCE ---")
    print(f"{'STEM':<6} | {'CANDIDATE NAME':<16} | {'TRUE GRADE':<11} | {'MAJ GRADE':<10} | {'GRADE MATCH':<12} | {'WIN ACC':<10} | {'SEG DET (+/-4s)'}")
    print("-" * 95)
    for c in per_candidate_results:
        g_match = "✔" if c["grade_match"] else "❌"
        seg_str = f"{c['segment_hits']}/{c['segment_total']} ({c['segment_acc']:4.1f}%)"
        print(f"{c['stem']:<6} | {c['name']:<16} | {c['true_grade']:<11} | {c['pred_majority_grade']:<10} | {g_match:<12} | {c['window_acc']:5.2f}%    | {seg_str}")

    print("=" * 105)
    print("KEY TAKEAWAYS FOR MODEL BENCHMARKING:")
    print(f"• The real empirical time-position floor is {overall_window_acc:.2f}% (NOT the fake 88.2% from synthetic labels).")
    print(f"• The majority-grade floor is {grade_matches}/{len(stems)} ({overall_grade_acc:.1f}%).")
    print("• Any future trained machine learning model MUST beat these numbers to prove real clinical value.")
    print("=" * 105)

    # Save to report
    report_data = {
        "evaluation_protocol": "Leave-One-Candidate-Out (LOSO) Cross-Validation",
        "ground_truth_source": "human_annotated (human_annotation_sheet.csv)",
        "cohort_size": len(stems),
        "metrics": {
            "time_position_baseline_window_accuracy": round(overall_window_acc, 2),
            "time_position_baseline_segment_accuracy_4s": round(overall_segment_acc, 2),
            "majority_grade_baseline_agreement": round(overall_grade_acc, 2),
            "majority_grade_correct_count": int(grade_matches),
            "majority_grade_total": len(stems),
        },
        "per_step_distribution": {
            sid: {
                "label": STEP_LABELS[sid],
                "done_count": len(step_positives[sid]),
                "missed_count": len(step_negatives[sid]),
                "missed_sessions": step_negatives[sid]
            } for sid in range(1, 14)
        },
        "candidate_results": per_candidate_results
    }

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nFull report written to: {OUT_REPORT}")

if __name__ == "__main__":
    run_phase2()
