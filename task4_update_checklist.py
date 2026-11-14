"""Task 4: Update Faculty OSCE Checklist Generator.

Requirements:
1. Replace start-end span with "first signal at mm:ss".
2. Add "Reliability" column (Low for steps 4, 6, 10; High for step 9; N/A for unverified).
3. Contradiction suppression: Never show FOUND for a step the faculty sheet contradicts
   in the evaluation set; show it as a known failure / false alarm.
"""

import os
import json
import csv
import numpy as np

from src.data_loader import STEP_LABELS

MANIFEST_PATH = r"C:\Medical\foley_cohort_manifest.csv"
FEATURE_DIR = r"C:\Medical\dataset\features_128px"
GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"
CHECKLIST_DIR = r"C:\Medical\reports\checklists"

def format_sec(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

def get_rule_detections(stem: str):
    feat_file = os.path.join(FEATURE_DIR, f"{stem}_feat.npz")
    if not os.path.exists(feat_file):
        return {}

    data = np.load(feat_file)
    X = data["X"]
    T = len(X)
    if T == 0:
        return {}

    grid_motion = X[:, 1:65]
    green_drape = X[:, 113]
    urine_yellow = X[:, 114]
    glove_white = X[:, 116]
    flux = X[:, 225] if X.shape[1] > 225 else np.zeros(T)

    detections = {}

    # Step 4: Glove White
    early_limit = max(3, int(T * 0.45))
    glove_early = glove_white[:early_limit]
    if len(glove_early) > 0:
        g_thresh = np.percentile(glove_early, 65)
        if (np.max(glove_early) - np.min(glove_early) > 0.015) and np.any(glove_early >= g_thresh):
            idxs = np.where(glove_early >= g_thresh)[0]
            detections[4] = {
                "first_signal_sec": float(idxs[0] * 2.0),
                "evidence": f"{np.max(glove_early)*100:.1f}% peak white reflection",
                "source": "Glove Colour Detector",
                "reliability": "Low (50% False-Alarm Rate)",
            }

    # Step 6: Green Drape
    d_thresh = np.percentile(green_drape, 65)
    if (np.max(green_drape) - np.min(green_drape) > 0.002) and np.any(green_drape >= d_thresh):
        idxs = np.where(green_drape >= d_thresh)[0]
        detections[6] = {
            "first_signal_sec": float(idxs[0] * 2.0),
            "evidence": f"{np.max(green_drape)*1000:.1f} per-mille green drape coverage",
            "source": "Green Drape Detector",
            "reliability": "Low (50% False-Alarm Rate on field movements)",
        }

    # Step 9: Yellow Fluid
    mid_start = int(T * 0.45)
    mid_end = int(T * 0.90)
    if mid_end > mid_start + 1:
        u_window = urine_yellow[mid_start:mid_end]
        if (np.max(u_window) - np.min(u_window) > 0.0025) and np.any(u_window >= np.percentile(u_window, 70)):
            idxs = np.where(u_window >= np.percentile(u_window, 70))[0]
            detections[9] = {
                "first_signal_sec": float((idxs[0] + mid_start) * 2.0),
                "evidence": f"{np.max(u_window)*1000:.1f} per-mille yellow urine flash",
                "source": "Urine Yellow Detector",
                "reliability": "High (0% False-Alarm Rate)",
            }

    # Step 10: Pelvic Motion (Syringe Proxy)
    if mid_end > mid_start + 2:
        pelvic = grid_motion[mid_start:mid_end, 24:48].mean(axis=1) + 0.1 * np.tanh(flux[mid_start:mid_end])
        if (np.max(pelvic) - np.min(pelvic) > 0.008) and np.any(pelvic >= np.percentile(pelvic, 75)):
            idxs = np.where(pelvic >= np.percentile(pelvic, 75))[0]
            detections[10] = {
                "first_signal_sec": float((idxs[0] + mid_start) * 2.0),
                "evidence": f"{np.max(pelvic)*100:.1f}% pelvic motion burst",
                "source": "Pelvic Motion Detector (Syringe Proxy)",
                "reliability": "Low (100% False-Alarm Rate on Step 10 Omissions)",
            }

    return detections

def generate_checklist(stem: str):
    with open(GT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    cand = cohort.get(stem, {})
    cand_name = cand.get("candidate_name", f"Resident-{stem}")
    fac_grade = cand.get("faculty_grade", "P")
    missing_steps = cand.get("missing_steps", [])

    rule_dets = get_rule_detections(stem)
    rows = []

    for sid in range(1, 14):
        sname = STEP_LABELS[sid]
        is_contradicted_by_faculty = (sid in missing_steps)

        if sid in rule_dets:
            det = rule_dets[sid]
            first_sig = format_sec(det["first_signal_sec"])
            rel = det["reliability"]

            if is_contradicted_by_faculty:
                # Contradiction: Detector fired, but faculty confirmed step was MISSED
                status = "KNOWN FAILURE / FALSE ALARM"
                evidence = f"Signal detected ({det['evidence']}), but contradicted by faculty ground truth (Resident missed step)"
                source = f"{det['source']} (Contradicted)"
            else:
                status = "FOUND (✔)"
                evidence = det["evidence"]
                source = det["source"]
        else:
            first_sig = "—"
            rel = "—"
            status = "NOT VERIFIED"
            evidence = "No detector signal"
            source = "Uninstrumented"

        rows.append({
            "sid": sid,
            "name": sname,
            "status": status,
            "first_sig": first_sig,
            "rel": rel,
            "evidence": evidence,
            "source": source
        })

    # Render Markdown table
    md = []
    md.append(f"# D Y PATIL MEDICAL COLLEGE & HOSPITAL")
    md.append(f"### Male Foley Catheterization OSCE - Verified Faculty Checklist")
    md.append(f"**Candidate:** Resident-{stem} ({cand_name}) | **Faculty Ground-Truth Grade:** {fac_grade}\n")
    md.append(f"> [!IMPORTANT]")
    md.append(f"> **Audit Policy:** Start-end spans replaced with single 'First Signal' timestamp. Detectors with known false-alarm issues (Steps 4, 6, 10) are explicitly marked **Low Reliability**. If faculty records a step as missed, any detector firing is flagged as a **KNOWN FAILURE / FALSE ALARM** rather than claiming success.\n")

    md.append(f"| Step | Clinical Step Name | AI Status | First Signal | Reliability | Measured Signal Evidence | Evidence Source | Faculty Confirmation |")
    md.append(f"| :---: | :--- | :---: | :---: | :---: | :--- | :--- | :---: |")

    for r in rows:
        if "KNOWN FAILURE" in r["status"]:
            stat_str = f"<span style='color:red;'>**{r['status']}**</span>"
        elif "FOUND" in r["status"]:
            stat_str = f"**{r['status']}**"
        else:
            stat_str = f"<span style='color:gray;'>{r['status']}</span>"

        md.append(f"| **{r['sid']:02d}** | {r['name']} | {stat_str} | `{r['first_sig']}` | {r['rel']} | {r['evidence']} | {r['source']} | [ ] Confirm<br>[ ] Overrule |")

    os.makedirs(CHECKLIST_DIR, exist_ok=True)
    out_path = os.path.join(CHECKLIST_DIR, f"Verified_Checklist_{stem}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return rows

if __name__ == "__main__":
    print("Generating updated checklists for spot-check videos (007, 011, 012, 016, 021)...")
    for s in ["007", "011", "012", "016", "021"]:
        rows = generate_checklist(s)
        # Check if contradiction was flagged
        contra = [r for r in rows if "KNOWN FAILURE" in r["status"]]
        print(f"Candidate {s}: {len(contra)} known failures/false alarms flagged. Checklist saved to {CHECKLIST_DIR}\\Verified_Checklist_{s}.md")
        for c in contra:
            print(f"  -> Flagged Step {c['sid']} ({c['name']}): First Signal at {c['first_sig']}, Reliability: {c['rel']}")
