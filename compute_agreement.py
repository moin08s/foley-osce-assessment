"""Inter-Annotator Agreement Computation Tool (Task 4).

Given two annotators' annotation sheets (or two annotator names within one sheet):
1. Calculates per-step Mean Absolute Difference (MAD) in start time (in seconds).
2. Calculates overall timing agreement (mean and standard deviation).
3. Reports the number and list of steps where annotators disagree on done/missed.
"""

import os
import sys
import csv
import argparse
from typing import Dict, List, Tuple
import numpy as np

from src.data_loader import STEP_LABELS

def parse_sheet(csv_path: str, target_annotator: str = None) -> Dict[Tuple[str, int], dict]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Annotation sheet not found: {csv_path}")

    records = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid = r.get("video", "").strip().upper()
            sid = int(r.get("step_id", 0))
            ann = r.get("annotator_name", "").strip()
            
            if target_annotator and ann.lower() != target_annotator.lower():
                continue

            status = r.get("status", "").strip().lower()
            # Default status to 'done' if start_sec is populated and not explicitly missed
            start_str = r.get("start_sec", "").strip()
            if not status:
                status = "done" if start_str else ""

            start_val = None
            if start_str:
                try:
                    start_val = float(start_str)
                except ValueError:
                    start_val = None

            end_str = r.get("end_sec", "").strip()
            end_val = None
            if end_str:
                try:
                    end_val = float(end_str)
                except ValueError:
                    end_val = None

            records[(vid, sid)] = {
                "video": vid,
                "step_id": sid,
                "step_label": r.get("step_label", STEP_LABELS.get(sid, f"Step_{sid}")),
                "status": status,
                "start_sec": start_val,
                "end_sec": end_val,
                "prompt_time_sec": r.get("prompt_time_sec", "").strip(),
                "prompt_step": r.get("prompt_step", "").strip(),
                "annotator_name": ann,
                "notes": r.get("notes", "").strip(),
            }
    return records


def compute_inter_annotator_agreement(sheet1_records: dict, sheet2_records: dict, name1="Annotator 1", name2="Annotator 2"):
    common_keys = sorted(list(set(sheet1_records.keys()) & set(sheet2_records.keys())))

    if not common_keys:
        print("Error: No overlapping (video, step_id) entries found between the two sets.")
        return

    print("=" * 105)
    print(f"INTER-ANNOTATOR AGREEMENT REPORT: {name1} vs {name2}")
    print(f"Total Overlapping Video-Step Items: {len(common_keys)}")
    print("=" * 105)

    # 1. Done / Missed Status Agreement
    status_agree = 0
    status_disagree = 0
    disagreements = []

    # 2. Timing Differences per Step
    step_timing_diffs = {s: [] for s in range(1, 14)}

    for k in common_keys:
        r1 = sheet1_records[k]
        r2 = sheet2_records[k]
        vid, sid = k

        st1 = r1["status"]
        st2 = r2["status"]

        # Only evaluate status if at least one annotator made a decision
        if st1 or st2:
            if st1 == st2:
                status_agree += 1
            else:
                status_disagree += 1
                disagreements.append({
                    "video": vid,
                    "step_id": sid,
                    "step_label": r1["step_label"],
                    "ann1_status": st1 or "(unmarked)",
                    "ann2_status": st2 or "(unmarked)",
                    "ann1_start": r1["start_sec"],
                    "ann2_start": r2["start_sec"],
                })

        # Timing difference (only if both recorded start_sec and both marked done)
        if r1["start_sec"] is not None and r2["start_sec"] is not None:
            diff = abs(r1["start_sec"] - r2["start_sec"])
            step_timing_diffs[sid].append(diff)

    # Print Status Agreement Summary
    total_status = status_agree + status_disagree
    pct_status = (status_agree / total_status * 100.0) if total_status > 0 else 0.0

    print("\n--- 1. STATUS AGREEMENT (DONE vs MISSED) ---")
    print(f"Total Evaluated Step Decisions:  {total_status}")
    print(f"Agreed Decisions:                {status_agree} ({pct_status:.1f}%)")
    print(f"Disagreed Decisions:             {status_disagree} ({100.0 - pct_status:.1f}%)")

    if disagreements:
        print("\nList of Status Disagreements:")
        print(f"{'VIDEO':<10} | {'STEP':<4} | {'STEP LABEL':<30} | {name1:<16} | {name2:<16}")
        print("-" * 85)
        for d in disagreements:
            print(f"{d['video']:<10} | {d['step_id']:<4} | {d['step_label']:<30} | {d['ann1_status']:<16} | {d['ann2_status']:<16}")
    else:
        print("✔ Zero status disagreements across all evaluated steps!")

    # Print Timing Differences Summary
    print("\n--- 2. TIMING AGREEMENT: MEAN ABSOLUTE DIFFERENCE (MAD) IN START SECONDS ---")
    print(f"{'STEP':<4} | {'STEP LABEL':<32} | {'PAIRS':<8} | {'MEAN ABS DIFF (s)':<20} | {'MEDIAN DIFF (s)':<18} | {'MAX DIFF (s)'}")
    print("-" * 105)

    all_diffs = []
    for sid in range(1, 14):
        diffs = step_timing_diffs[sid]
        slabel = STEP_LABELS.get(sid, f"Step_{sid}")
        if diffs:
            mean_d = float(np.mean(diffs))
            med_d = float(np.median(diffs))
            max_d = float(np.max(diffs))
            all_diffs.extend(diffs)
            print(f"{sid:<4} | {slabel:<32} | {len(diffs):<8} | {mean_d:<20.2f} | {med_d:<18.2f} | {max_d:<10.2f}")
        else:
            print(f"{sid:<4} | {slabel:<32} | {'0':<8} | {'N/A (No paired times)':<20} | {'—':<18} | {'—'}")

    print("-" * 105)
    if all_diffs:
        overall_mean = float(np.mean(all_diffs))
        overall_std = float(np.std(all_diffs))
        overall_med = float(np.median(all_diffs))
        print(f"OVERALL TIMING MAD (All Steps Combined): {overall_mean:.2f} s ± {overall_std:.2f} s (Median: {overall_med:.2f} s, N={len(all_diffs)} pairs)")
    else:
        print("OVERALL TIMING MAD: No pairs with recorded start times to compare.")
    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="Compute agreement between two annotation sets.")
    parser.add_argument("--sheet1", type=str, default=r"C:\Medical\human_annotation_sheet.csv", help="Path to first CSV")
    parser.add_argument("--sheet2", type=str, default=None, help="Path to second CSV (optional if single sheet has multiple annotators)")
    parser.add_argument("--ann1", type=str, default=None, help="Name of annotator 1")
    parser.add_argument("--ann2", type=str, default=None, help="Name of annotator 2")
    args = parser.parse_args()

    if args.sheet2:
        r1 = parse_sheet(args.sheet1, args.ann1)
        r2 = parse_sheet(args.sheet2, args.ann2)
        n1 = args.ann1 or os.path.basename(args.sheet1)
        n2 = args.ann2 or os.path.basename(args.sheet2)
    else:
        # Single sheet with two annotators
        if not args.ann1 or not args.ann2:
            # Detect annotators in single sheet
            with open(args.sheet1, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                names = sorted(list(set(r.get("annotator_name", "").strip() for r in reader if r.get("annotator_name", "").strip())))
            if len(names) >= 2:
                args.ann1 = names[0]
                args.ann2 = names[1]
                print(f"Auto-detected annotators in {args.sheet1}: '{args.ann1}' and '{args.ann2}'")
            else:
                print(f"Sheet contains annotators: {names}. Provide --sheet2 or --ann1 and --ann2.")
                sys.exit(1)
        r1 = parse_sheet(args.sheet1, args.ann1)
        r2 = parse_sheet(args.sheet1, args.ann2)
        n1 = args.ann1
        n2 = args.ann2

    compute_inter_annotator_agreement(r1, r2, n1, n2)

if __name__ == "__main__":
    main()
