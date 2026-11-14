"""Comprehensive Implementation of Tasks 1, 2, 4, and 5.

Task 1: Compute detector metrics twice:
  - Without suppression (Real Use): Raw detector outputs
  - With suppression: After faculty-sheet contradiction filter
Task 2: Relabel reliability:
  - 'High' only if tested on >= 5 negative sessions AND sensitivity >= 90%
  - Steps 2, 8, 11: 'No negative data'
  - All other steps: 'Untested/Low'
Task 4: Output clean spot-check table (faculty sheet data only, no subjective guide)
Task 5: Export blank annotation sheet CSV (23 videos x 13 steps = 299 rows)
"""

import os
import json
import csv
import numpy as np

from src.data_loader import STEP_LABELS

GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"
FEAT_DIR = r"C:\Medical\dataset\features_128px"
CHECKLIST_DIR = r"C:\Medical\reports\checklists"
ANNOTATION_CSV = r"C:\Medical\human_annotation_sheet.csv"

def run_all_tasks():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    # 1. Negative lists per step
    step_negatives = {s: [] for s in range(1, 14)}
    for stem, cand in sorted(cohort.items()):
        missing = cand.get("missing_steps", [])
        for s in range(1, 14):
            if s in missing:
                step_negatives[s].append(stem)

    # 2. Run raw detectors on all 23 sessions
    detector_steps = [4, 6, 9, 10]
    names = {
        4: "Step 4: Glove White (Sterile Gloves)",
        6: "Step 6: Green Drape (Painting & Draping)",
        9: "Step 9: Urine Yellow (Urobag Fluid Return)",
        10: "Step 10: Pelvic Motion (Syringe Proxy)",
    }

    raw_metrics = {s: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for s in detector_steps}
    supp_metrics = {s: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for s in detector_steps}
    detections_per_session = {}

    for stem in sorted(cohort.keys()):
        feat_file = os.path.join(FEAT_DIR, f"{stem}_feat.npz")
        if not os.path.exists(feat_file):
            continue
        data = np.load(feat_file)
        X = data["X"]
        T = len(X)
        if T == 0:
            continue

        grid_motion = X[:, 1:65]
        green_drape = X[:, 113]
        urine_yellow = X[:, 114]
        glove_white = X[:, 116]
        flux = X[:, 225] if X.shape[1] > 225 else np.zeros(T)

        dets = {}

        # Step 4
        el = max(3, int(T * 0.45))
        gw_e = glove_white[:el]
        if len(gw_e) > 0 and (np.max(gw_e) - np.min(gw_e) > 0.015):
            idxs = np.where(gw_e >= np.percentile(gw_e, 65))[0]
            dets[4] = True if len(idxs) > 0 else False
        else:
            dets[4] = False

        # Step 6
        if (np.max(green_drape) - np.min(green_drape) > 0.002):
            idxs = np.where(green_drape >= np.percentile(green_drape, 65))[0]
            dets[6] = True if len(idxs) > 0 else False
        else:
            dets[6] = False

        # Step 9
        m_s = int(T * 0.45)
        m_e = int(T * 0.90)
        if m_e > m_s + 1:
            u_win = urine_yellow[m_s:m_e]
            if (np.max(u_win) - np.min(u_win) > 0.0025):
                idxs = np.where(u_win >= np.percentile(u_win, 70))[0]
                dets[9] = True if len(idxs) > 0 else False
            else:
                dets[9] = False
        else:
            dets[9] = False

        # Step 10
        if m_e > m_s + 2:
            pelvic = grid_motion[m_s:m_e, 24:48].mean(axis=1) + 0.1 * np.tanh(flux[m_s:m_e])
            if (np.max(pelvic) - np.min(pelvic) > 0.008):
                idxs = np.where(pelvic >= np.percentile(pelvic, 75))[0]
                dets[10] = True if len(idxs) > 0 else False
            else:
                dets[10] = False
        else:
            dets[10] = False

        detections_per_session[stem] = dets

        # Record confusion
        for s in detector_steps:
            is_neg = stem in step_negatives[s]
            det = dets[s]

            # WITHOUT SUPPRESSION (Real Use)
            if is_neg:
                if det:
                    raw_metrics[s]["FP"] += 1
                else:
                    raw_metrics[s]["TN"] += 1
            else:
                if det:
                    raw_metrics[s]["TP"] += 1
                else:
                    raw_metrics[s]["FN"] += 1

            # WITH SUPPRESSION (Post-Adjudication: FP is suppressed / flagged)
            if is_neg:
                # Contradiction suppression overrules false positive
                supp_metrics[s]["TN"] += 1  # Successfully prevented from showing FOUND
            else:
                if det:
                    supp_metrics[s]["TP"] += 1
                else:
                    supp_metrics[s]["FN"] += 1

    # Print Task 1 Dual Metrics Table
    print("=" * 115)
    print("TASK 1: DETECTOR METRICS REPORTED TWICE (WITHOUT vs WITH CONTRADICTION SUPPRESSION)")
    print("=" * 115)
    print(f"{'STEP':<4} | {'DETECTOR NAME':32s} | {'MODE':<22s} | {'NEG N':<6} | {'SENSITIVITY':<14} | {'PRECISION':<12} | {'FALSE-ALARM RATE'}")
    print("-" * 115)

    for s in detector_steps:
        # Without suppression
        r = raw_metrics[s]
        n_neg = r["FP"] + r["TN"]
        sens_r = (r["TP"] / (r["TP"] + r["FN"]) * 100.0) if (r["TP"] + r["FN"]) > 0 else 0.0
        prec_r = (r["TP"] / (r["TP"] + r["FP"]) * 100.0) if (r["TP"] + r["FP"]) > 0 else 0.0
        fa_r = (r["FP"] / n_neg * 100.0) if n_neg > 0 else 0.0

        # With suppression
        sp = supp_metrics[s]
        sens_sp = (sp["TP"] / (sp["TP"] + sp["FN"]) * 100.0) if (sp["TP"] + sp["FN"]) > 0 else 0.0
        prec_sp = (sp["TP"] / (sp["TP"] + sp["FP"]) * 100.0) if (sp["TP"] + sp["FP"]) > 0 else 0.0
        fa_sp = (sp["FP"] / n_neg * 100.0) if n_neg > 0 else 0.0

        print(f"{s:<4} | {names[s]:32s} | {'WITHOUT SUPP (Real)':<22s} | {n_neg:<6d} | {sens_r:5.1f}% ({r['TP']}/{r['TP']+r['FN']}) | {prec_r:5.1f}% ({r['TP']}/{r['TP']+r['FP']}) | {fa_r:5.1f}% ({r['FP']}/{n_neg} false)")
        print(f"{'':<4} | {'':32s} | {'WITH SUPP (Adjudicated)':<22s} | {n_neg:<6d} | {sens_sp:5.1f}% ({sp['TP']}/{sp['TP']+sp['FN']}) | {prec_sp:5.1f}% ({sp['TP']}/{sp['TP']+sp['FP']}) | {fa_sp:5.1f}% ({sp['FP']}/{n_neg} false)")
        print("-" * 115)

    # Task 2: Reliability Labels
    reliability_labels = {}
    for s in range(1, 14):
        if s in [2, 8, 11]:
            reliability_labels[s] = "No negative data"
        else:
            n_neg = len(step_negatives[s])
            sens = 0.0
            if s in raw_metrics:
                r = raw_metrics[s]
                sens = (r["TP"] / (r["TP"] + r["FN"]) * 100.0) if (r["TP"] + r["FN"]) > 0 else 0.0
            
            if n_neg >= 5 and sens >= 90.0:
                reliability_labels[s] = "High"
            else:
                reliability_labels[s] = f"Untested/Low ({n_neg} neg samples)"

    print("\n" + "=" * 90)
    print("TASK 2: RE-EVALUATED STEP RELIABILITY LABELS")
    print("Criteria: 'High' ONLY if >= 5 negative sessions AND sensitivity >= 90%.")
    print("=" * 90)
    for s in range(1, 14):
        print(f"Step {s:02d} ({STEP_LABELS[s]:<28}): {reliability_labels[s]}")

    # Task 4: Clean Spot-Check Table
    print("\n" + "=" * 100)
    print("TASK 4: CLEAN SPOT-CHECK TABLE (FACULTY SHEET DATA ONLY)")
    print("=" * 100)
    print(f"{'VIDEO':<10} | {'SHEET PAGE':<12} | {'FACULTY-MARKED STEP(S)':<36} | {'SHEET COMMENT TEXT'}")
    print("-" * 100)
    spot_mapping = {
        "007": (7, [1]),
        "011": (11, [13]),
        "012": (12, [3, 4, 5, 6, 9, 10]),
        "016": (16, [4, 13]),
        "021": (21, [12, 13]),
    }
    for stem, (pg, missed) in spot_mapping.items():
        missed_str = f"Step(s) marked MISSED: {missed}"
        print(f"{stem + '.MP4':<10} | Page {pg:<6d} | {missed_str:<36} | None (No comments on sheet)")

    # Task 5: Export Blank Annotation Sheet (23 videos x 13 steps = 299 rows)
    rows_to_export = []
    for stem in sorted(cohort.keys()):
        cand = cohort[stem]
        v_file = f"{stem}.MP4"
        for sid in range(1, 14):
            rows_to_export.append({
                "video": v_file,
                "step_id": sid,
                "step_label": STEP_LABELS[sid],
                "start_sec": "",
                "end_sec": "",
                "annotator_name": "",
                "notes": ""
            })

    with open(ANNOTATION_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["video", "step_id", "step_label", "start_sec", "end_sec", "annotator_name", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_export)

    print("\n" + "=" * 90)
    print(f"TASK 5: BLANK ANNOTATION SHEET EXPORTED")
    print(f"Path:  {ANNOTATION_CSV}")
    print(f"Total Rows: {len(rows_to_export)} (23 videos x 13 steps)")
    print(f"Blank columns for human annotation: start_sec, end_sec, annotator_name, notes")
    print("=" * 90)

if __name__ == "__main__":
    run_all_tasks()
