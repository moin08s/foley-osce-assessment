"""Initialize human_annotation_sheet.csv with new columns:
video,step_id,step_label,status,start_sec,end_sec,prompt_time_sec,prompt_step,annotator_name,notes
"""

import os
import csv
import json

from src.data_loader import STEP_LABELS

GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"
ANNOTATION_CSV = r"C:\Medical\human_annotation_sheet.csv"

def init_schema():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    stems = sorted(list(cohort.keys()))
    fieldnames = [
        "video",
        "step_id",
        "step_label",
        "status",
        "start_sec",
        "end_sec",
        "prompt_time_sec",
        "prompt_step",
        "annotator_name",
        "notes"
    ]

    rows = []
    for s in stems:
        v_file = f"{s}.MP4"
        for sid in range(1, 14):
            rows.append({
                "video": v_file,
                "step_id": str(sid),
                "step_label": STEP_LABELS[sid],
                "status": "",
                "start_sec": "",
                "end_sec": "",
                "prompt_time_sec": "",
                "prompt_step": "",
                "annotator_name": "",
                "notes": ""
            })

    with open(ANNOTATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Initialized {ANNOTATION_CSV} with new schema. Total rows: {len(rows)}")

if __name__ == "__main__":
    init_schema()
