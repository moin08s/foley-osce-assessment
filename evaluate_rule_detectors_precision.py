"""Evaluate Rule-Based Detectors on Absent/Incomplete Steps (Instruction 4).

Evaluates precision, false-alarm rate (fall-out), specificity, and sensitivity
across sessions where the step was omitted or incomplete according to faculty.

Evaluated Steps:
- Step 4: Glove Colour / Hand Hygiene (Missed in sessions: 012, 013, 016, 023)
- Step 6: Green Drape (Missed in sessions: 009, 012)
- Step 9: Yellow Fluid in Urobag Tube (Missed in sessions: 012)
- Step 10: Syringe at Side Port / Balloon (Missed in sessions: 012)

Pure Python + NumPy ONLY.
"""

import os
import json
import numpy as np

from src.data_loader import STEP_LABELS

def evaluate_detector_precision_and_false_alarms():
    feat_dir = r"C:\Medical\dataset\features_128px"
    gt_path = r"C:\Medical\dataset\cohort_ground_truth.json"

    with open(gt_path, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    steps_to_evaluate = [4, 6, 9, 10]
    names = {
        4: "Step 4: Glove Colour (Sterile Gloves)",
        6: "Step 6: Green Drape (Painting & Draping)",
        9: "Step 9: Yellow Fluid (Urobag & Urine Return)",
        10: "Step 10: Syringe at Port (Balloon Inflation)",
    }

    metrics = {
        s: {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "neg_sessions": [], "pos_sessions": []}
        for s in steps_to_evaluate
    }

    for stem in sorted(cohort.keys()):
        cand = cohort[stem]
        missing_steps = cand.get("missing_steps", [])

        feat_file = os.path.join(feat_dir, f"{stem}_feat.npz")
        if not os.path.exists(feat_file):
            continue

        data = np.load(feat_file)
        X = data["X"]
        T = len(X)
        if T == 0:
            continue

        # Feature signals
        grid_motion = X[:, 1:65]
        green_drape = X[:, 113]
        urine_yellow = X[:, 114]
        glove_white = X[:, 116]
        flux = X[:, 225] if X.shape[1] > 225 else np.zeros(T)

        detections = {}

        # 1. Step 4 Glove Detector
        # High-confidence threshold
        early_limit = max(3, int(T * 0.45))
        glove_early = glove_white[:early_limit]
        if len(glove_early) > 0:
            g_thresh = np.percentile(glove_early, 65)
            # Must exceed baseline by significant absolute delta
            if (np.max(glove_early) - np.min(glove_early) > 0.015) and np.any(glove_early >= g_thresh):
                detections[4] = True
            else:
                detections[4] = False
        else:
            detections[4] = False

        # 2. Step 6 Green Drape Detector
        d_thresh = np.percentile(green_drape, 65)
        if (np.max(green_drape) - np.min(green_drape) > 0.002) and np.any(green_drape >= d_thresh):
            detections[6] = True
        else:
            detections[6] = False

        # 3. Step 9 Urine Yellow Detector
        mid_start = int(T * 0.45)
        mid_end = int(T * 0.90)
        if mid_end > mid_start + 1:
            u_window = urine_yellow[mid_start:mid_end]
            if (np.max(u_window) - np.min(u_window) > 0.0025) and np.any(u_window >= np.percentile(u_window, 70)):
                detections[9] = True
            else:
                detections[9] = False
        else:
            detections[9] = False

        # 4. Step 10 Syringe / Bulb Detector
        if mid_end > mid_start + 2:
            pelvic = grid_motion[mid_start:mid_end, 24:48].mean(axis=1) + 0.5 * flux[mid_start:mid_end]
            if (np.max(pelvic) - np.min(pelvic) > 0.008) and np.any(pelvic >= np.percentile(pelvic, 75)):
                detections[10] = True
            else:
                detections[10] = False
        else:
            detections[10] = False

        # Record confusion for each step
        for s in steps_to_evaluate:
            is_present = (s not in missing_steps)
            is_detected = detections[s]

            if is_present:
                metrics[s]["pos_sessions"].append(stem)
                if is_detected:
                    metrics[s]["TP"] += 1
                else:
                    metrics[s]["FN"] += 1
            else:
                metrics[s]["neg_sessions"].append(stem)
                if is_detected:
                    metrics[s]["FP"] += 1
                else:
                    metrics[s]["TN"] += 1

    print("=" * 115)
    print("INSTRUCTION 4: RULE-BASED DETECTOR EVALUATION ON ABSENT / INCOMPLETE SESSIONS")
    print("=" * 115)
    print(f"{'STEP':4s} | {'DETECTOR NAME':32s} | {'TP':4s} | {'FP':4s} | {'TN':4s} | {'FN':4s} | {'PRECISION':10s} | {'SENSITIVITY':12s} | {'FALSE ALARM RATE'}")
    print("-" * 115)

    for s in steps_to_evaluate:
        d = metrics[s]
        tp = d["TP"]
        fp = d["FP"]
        tn = d["TN"]
        fn = d["FN"]

        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        sens = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        fa_rate = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0

        print(f"{s:4d} | {names[s]:32s} | {tp:4d} | {fp:4d} | {tn:4d} | {fn:4d} | {prec:9.1f}% | {sens:11.1f}% | {fa_rate:9.1f}% ({fp}/{fp+tn})")
        if d["neg_sessions"]:
            print(f"     Negative sessions (omitted by resident): {d['neg_sessions']}")

    print("-" * 115)
    return metrics


if __name__ == "__main__":
    evaluate_detector_precision_and_false_alarms()
