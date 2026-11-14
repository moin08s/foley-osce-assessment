"""Update Faculty OSCE Checklists with dual metrics and new reliability labels."""

import os
import json
import numpy as np

from src.data_loader import STEP_LABELS

GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"
FEATURE_DIR = r"C:\Medical\dataset\features_128px"
CHECKLIST_DIR = r"C:\Medical\reports\checklists"

def format_sec(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

# Reliability under new criteria
RELIABILITY = {
    1: "Untested/Low (1 neg sample)",
    2: "No negative data",
    3: "Untested/Low (2 neg samples)",
    4: "Untested/Low (4 neg samples, 50.0% False-Alarm Rate)",
    5: "Untested/Low (3 neg samples)",
    6: "Untested/Low (2 neg samples, 50.0% False-Alarm Rate)",
    7: "Untested/Low (1 neg sample)",
    8: "No negative data",
    9: "Untested/Low (1 neg sample, 54.5% Sensitivity)",
    10: "Untested/Low (1 neg sample, 100.0% False-Alarm Rate)",
    11: "No negative data",
    12: "Untested/Low (2 neg samples)",
    13: "Untested/Low (3 neg samples)",
}

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
            }

    # Step 6: Green Drape
    d_thresh = np.percentile(green_drape, 65)
    if (np.max(green_drape) - np.min(green_drape) > 0.002) and np.any(green_drape >= d_thresh):
        idxs = np.where(green_drape >= d_thresh)[0]
        detections[6] = {
            "first_signal_sec": float(idxs[0] * 2.0),
            "evidence": f"{np.max(green_drape)*1000:.1f} per-mille green drape coverage",
            "source": "Green Drape Detector",
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
        rel = RELIABILITY[sid]
        is_contradicted_by_faculty = (sid in missing_steps)

        if sid in rule_dets:
            det = rule_dets[sid]
            first_sig = format_sec(det["first_signal_sec"])

            if is_contradicted_by_faculty:
                status = "KNOWN FAILURE / FALSE ALARM (Suppressed)"
                evidence = f"Signal detected ({det['evidence']}), but contradicted by faculty ground truth (Resident missed step)"
                source = f"{det['source']} (Contradicted)"
            else:
                status = "FOUND (✔)"
                evidence = det["evidence"]
                source = det["source"]
        else:
            first_sig = "—"
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

    md = []
    md.append(f"# D Y PATIL MEDICAL COLLEGE & HOSPITAL")
    md.append(f"### Male Foley Catheterization OSCE - Verified Faculty Checklist")
    md.append(f"**Candidate:** Resident-{stem} ({cand_name}) | **Faculty Ground-Truth Grade:** {fac_grade}\n")
    md.append(f"> [!IMPORTANT]")
    md.append(f"> **Audit & Evaluation Policy:**")
    md.append(f"> 1. **Detector Metrics (Reported Twice):**")
    md.append(f">    - **Real Use (Without Contradiction Suppression):** Step 4 False-Alarm = 50.0% (2/4), Step 6 False-Alarm = 50.0% (1/2), Step 9 False-Alarm = 0.0% (0/1), Step 10 False-Alarm = 100.0% (1/1).")
    md.append(f">    - **With Contradiction Suppression:** Apparent False-Alarm = 0.0% across all steps because the faculty sheet overrides false alarms.")
    md.append(f"> 2. **Reliability Definition:** 'High' requires testing on >= 5 negative sessions AND sensitivity >= 90%. Steps 2, 8, 11 have 'No negative data'. All other steps are 'Untested/Low'.\n")

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
    for s in ["007", "011", "012", "016", "021"]:
        generate_checklist(s)
    print("Checklists updated with dual metrics and strict reliability criteria.")
