"""Compare all human annotations against faculty sheets across all 23 sessions."""

import os
import json
import csv

BASE_DIR = r"C:\Medical"
GT_PATH = os.path.join(BASE_DIR, "dataset", "cohort_ground_truth.json")
ANN_A_PATH = os.path.join(BASE_DIR, "human_annotation_sheet_Annotator_A.csv")

def compare():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    # Read Annotator A's decisions
    ann_decisions = {}
    with open(ANN_A_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid = r["video"].replace(".MP4", "").replace(".mp4", "")
            sid = int(r["step_id"])
            if vid not in ann_decisions:
                ann_decisions[vid] = {"missed": []}
            if r["status"].lower() == "missed":
                ann_decisions[vid]["missed"].append(sid)

    print("=" * 105)
    print("PHASE 1: COMPARISON OF HUMAN ANNOTATIONS AGAINST FACULTY SHEETS (ALL 23 SESSIONS)")
    print("=" * 105)
    print(f"{'STEM':<6} | {'CANDIDATE':<16} | {'ANNOTATOR MISSED':<22} | {'FACULTY MISSED':<22} | {'ALIGNMENT'}")
    print("-" * 105)

    perfect_match_count = 0
    mismatches = []

    for stem in sorted(cohort.keys()):
        cand = cohort[stem]
        cname = cand.get("candidate_name", stem)
        fac_missed = sorted(cand.get("missing_steps", []))
        ann_missed = sorted(ann_decisions.get(stem, {}).get("missed", []))

        is_match = (ann_missed == fac_missed)
        if is_match:
            perfect_match_count += 1
            align_str = "✔ EXACT MATCH"
        else:
            align_str = "❌ MISMATCH"
            mismatches.append({
                "stem": stem,
                "candidate": cname,
                "ann_missed": ann_missed,
                "fac_missed": fac_missed
            })

        ann_str = str(ann_missed) if ann_missed else "None (All done)"
        fac_str = str(fac_missed) if fac_missed else "None (All done)"
        print(f"{stem:<6} | {cname:<16} | {ann_str:<22} | {fac_str:<22} | {align_str}")

    print("-" * 105)
    print(f"Total Sessions Evaluated: {len(cohort)}")
    print(f"Sessions with Perfect Agreement: {perfect_match_count} / {len(cohort)} ({perfect_match_count/len(cohort)*100:.1f}%)")
    print(f"Sessions with Mismatches: {len(mismatches)} / {len(cohort)}")

    if mismatches:
        print("\nMismatch Details:")
        for m in mismatches:
            print(f"  Candidate {m['stem']} ({m['candidate']}): Annotator marked missed {m['ann_missed']}, Faculty marked missed {m['fac_missed']}")
    else:
        print("\n✔ All human annotations are 100% aligned with official faculty ground truth!")

    print("=" * 105)

if __name__ == "__main__":
    compare()
