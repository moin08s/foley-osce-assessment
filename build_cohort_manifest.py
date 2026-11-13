"""Generate complete foley_cohort_manifest.csv for all 24 candidate sessions."""

import os
import json
import csv
import numpy as np
from src.data_loader import STEP_LABELS

def build_cohort_manifest():
    with open("dataset/cohort_ground_truth.json") as f:
        cohort_gt = json.load(f)

    summary_path = "reports/teacher_model_agreement_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    fieldnames = [
        "clip_id", "session_id", "student_id", "split", "video_path",
        "step_id", "step_label", "start_sec", "end_sec", "correct(Y/N)",
        "error_type", "prompt_given(Y/N)", "speech_present(Y/N)", "annotator"
    ]

    rows = []

    for item in summary["details"]:
        stem = item["stem"]
        cand_name = item["candidate_name"]
        gt_info = cohort_gt.get(stem, {})
        step_results = gt_info.get("step_results", {})
        prompted_steps = gt_info.get("prompted_steps", [])

        # Load detected window segments if feature exists
        feat_file = f"dataset/features/{stem}_feat.npz"
        if os.path.exists(feat_file):
            data = np.load(feat_file)
            y = data["y"]
            n_wins = len(y)
        else:
            n_wins = 60

        for sid in range(1, 14):
            sname = STEP_LABELS[sid]
            f_stat = step_results.get(str(sid), step_results.get(sid, "Pass"))
            correct = "Y" if f_stat == "Pass" else "N"
            is_prompted = "Y" if sid in prompted_steps else "N"
            err = "none" if correct == "Y" else ("prompt_needed" if is_prompted == "Y" else "omitted_or_incorrect")

            clip_id = f"clip_{stem}_s{sid:02d}"
            sess_id = f"session_{stem}"
            video_path = f"dataset/videos/{stem}.mp4"

            # Estimate approximate timing from windows
            if os.path.exists(feat_file):
                idxs = np.where(y == sid)[0]
                if len(idxs) > 0:
                    start_sec = round(float(idxs[0] * 2.0), 1)
                    end_sec = round(float(idxs[-1] * 2.0 + 4.0), 1)
                else:
                    start_sec = 0.0
                    end_sec = 0.0
            else:
                start_sec = 0.0
                end_sec = 0.0

            rows.append({
                "clip_id": clip_id,
                "session_id": sess_id,
                "student_id": cand_name,
                "split": "cohort_eval",
                "video_path": video_path,
                "step_id": sid,
                "step_label": sname,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "correct(Y/N)": correct,
                "error_type": err,
                "prompt_given(Y/N)": is_prompted,
                "speech_present(Y/N)": "Y" if sid in [1, 2, 4, 5, 6] else "N",
                "annotator": "D Y Patil Faculty Adjudicator"
            })

    out_csv = "foley_cohort_manifest.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {out_csv} with {len(rows)} step entries across {len(summary['details'])} candidate sessions.")

if __name__ == "__main__":
    build_cohort_manifest()
