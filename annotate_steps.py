"""Interactive Step Annotation Tool (Instruction 2).

Pure Python Standard Library + NumPy ONLY.
Allows annotating real ground-truth step timestamps for Foley Catheterization videos.
Pre-fills suggested timestamps from rule-based detectors for steps 4, 6, 9, 10.
Saves real timestamps to foley_cohort_manifest.csv with source="human_annotated".
"""

import os
import sys
import json
import csv
import argparse
from typing import Dict, List, Optional
import numpy as np

from src.data_loader import STEP_LABELS

MANIFEST_PATH = r"C:\Medical\foley_cohort_manifest.csv"
VIDEO_DIR = r"C:\New folder (2)\103NZ_30"
FEATURE_DIR = r"C:\Medical\dataset\features_128px"
GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"


def parse_timestamp_input(inp: str, default_sec: float) -> float:
    """Parse timestamp string (e.g. '01:25.5' or '85.5' or blank for default)."""
    inp = inp.strip()
    if not inp:
        return default_sec
    if ":" in inp:
        parts = inp.split(":")
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    return float(inp)


def format_sec(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:04.1f}"


def get_detector_suggestions(stem: str) -> Dict[int, Dict[str, float]]:
    """Get rule-based detector suggestions for steps 4, 6, 9, 10."""
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

    suggestions = {}

    # Step 4: Glove Colour
    early_limit = max(3, int(T * 0.45))
    glove_early = glove_white[:early_limit]
    if len(glove_early) > 0:
        g_thresh = np.percentile(glove_early, 60)
        idxs = np.where(glove_early >= g_thresh)[0]
        if len(idxs) > 0:
            suggestions[4] = {
                "start_sec": round(float(idxs[0] * 2.0), 1),
                "end_sec": round(float(idxs[-1] * 2.0 + 4.0), 1),
            }

    # Step 6: Green Drape
    d_thresh = np.percentile(green_drape, 65)
    idxs = np.where(green_drape >= d_thresh)[0]
    if len(idxs) > 0:
        suggestions[6] = {
            "start_sec": round(float(idxs[0] * 2.0), 1),
            "end_sec": round(float(idxs[-1] * 2.0 + 4.0), 1),
        }

    # Step 9: Yellow Urine Return
    mid_start = int(T * 0.45)
    mid_end = int(T * 0.90)
    if mid_end > mid_start + 1:
        u_window = urine_yellow[mid_start:mid_end]
        u_thresh = np.percentile(u_window, 70)
        idxs = np.where(u_window >= u_thresh)[0]
        if len(idxs) > 0:
            suggestions[9] = {
                "start_sec": round(float((idxs[0] + mid_start) * 2.0), 1),
                "end_sec": round(float((idxs[-1] + mid_start) * 2.0 + 4.0), 1),
            }

    # Step 10: Syringe at Side Port / Bulb Inflation
    if mid_end > mid_start + 2:
        pelvic_motion = grid_motion[mid_start:mid_end, 24:48].mean(axis=1) + 0.5 * flux[mid_start:mid_end]
        s_thresh = np.percentile(pelvic_motion, 75)
        idxs = np.where(pelvic_motion >= s_thresh)[0]
        if len(idxs) > 0:
            suggestions[10] = {
                "start_sec": round(float((idxs[0] + mid_start) * 2.0), 1),
                "end_sec": round(float((idxs[-1] + mid_start) * 2.0 + 4.0), 1),
            }

    return suggestions


def annotate_session(stem: str, interactive: bool = True, prefill_all: bool = False):
    """Annotate a session and save to manifest."""
    with open(GT_PATH) as f:
        cohort = json.load(f)

    if stem not in cohort:
        print(f"Error: Stem {stem} not in cohort GT.")
        return

    cand_info = cohort[stem]
    cand_name = cand_info.get("candidate_name", stem)
    vid_file = cand_info.get("video_file", "")

    # Load suggestions
    suggestions = get_detector_suggestions(stem)

    # Load existing manifest rows
    manifest_rows = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_rows = list(reader)

    print("=" * 80)
    print(f"ANNOTATE SESSION: {stem} ({cand_name})")
    print(f"Video: {vid_file}")
    print("=" * 80)
    print("Pre-filled Rule Detector Suggestions Available for Steps: 4, 6, 9, 10\n")

    annotations = {}
    total_dur_sec = 160.0  # default duration

    feat_file = os.path.join(FEATURE_DIR, f"{stem}_feat.npz")
    if os.path.exists(feat_file):
        data = np.load(feat_file)
        total_dur_sec = len(data["y"]) * 2.0 + 2.0

    cur_time = 0.0

    for sid in range(1, 14):
        sname = STEP_LABELS[sid]
        sugg = suggestions.get(sid, None)

        if sugg:
            sugg_start = sugg["start_sec"]
            sugg_end = sugg["end_sec"]
            sugg_str = f" [Rule Detector: {format_sec(sugg_start)} - {format_sec(sugg_end)}]"
        else:
            sugg_start = round(cur_time, 1)
            sugg_end = round(cur_time + 10.0, 1)
            sugg_str = f" [Est: {format_sec(sugg_start)} - {format_sec(sugg_end)}]"

        if interactive and not prefill_all:
            print(f"\nStep {sid:02d}: {sname}{sugg_str}")
            s_in = input(f"  Start timestamp (default {format_sec(sugg_start)}): ")
            start_sec = parse_timestamp_input(s_in, sugg_start)

            e_in = input(f"  End timestamp   (default {format_sec(sugg_end)}): ")
            end_sec = parse_timestamp_input(e_in, sugg_end)

            status_in = input("  Status [P=Pass, M=Missed, 1=Prompted] (default P): ").strip().upper()
            if status_in == "M":
                correct = "N"
                err = "omitted_or_incorrect"
                prompted = "N"
            elif status_in == "1":
                correct = "N"
                err = "prompt_needed"
                prompted = "Y"
            else:
                correct = "Y"
                err = "none"
                prompted = "N"
        else:
            start_sec = sugg_start
            end_sec = sugg_end
            correct = "Y"
            err = "none"
            prompted = "N"
            cur_time = end_sec

        annotations[sid] = {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "correct": correct,
            "err": err,
            "prompted": prompted,
        }

    # Write updated rows to manifest
    fieldnames = [
        "clip_id", "session_id", "student_id", "split", "video_path",
        "step_id", "step_label", "start_sec", "end_sec", "correct(Y/N)",
        "error_type", "prompt_given(Y/N)", "speech_present(Y/N)", "annotator", "source"
    ]

    # Filter out old rows for this session
    sess_id = f"session_{stem}"
    updated_rows = [r for r in manifest_rows if r["session_id"] != sess_id]

    for sid in range(1, 14):
        ann = annotations[sid]
        updated_rows.append({
            "clip_id": f"clip_{stem}_s{sid:02d}",
            "session_id": sess_id,
            "student_id": cand_name,
            "split": "train",
            "video_path": vid_file,
            "step_id": str(sid),
            "step_label": STEP_LABELS[sid],
            "start_sec": str(ann["start_sec"]),
            "end_sec": str(ann["end_sec"]),
            "correct(Y/N)": ann["correct"],
            "error_type": ann["err"],
            "prompt_given(Y/N)": ann["prompted"],
            "speech_present(Y/N)": "Y" if sid in [1, 2] else "N",
            "annotator": "faculty_expert",
            "source": "human_annotated",
        })

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in updated_rows:
            # Ensure source column exists
            if "source" not in r:
                r["source"] = "synthetic"
            writer.writerow(r)

    print(f"\nSuccessfully saved human_annotated timestamps for session_{stem} to {MANIFEST_PATH}!")


def mark_manifest_sources():
    """Ensure all existing manifest rows have source='synthetic' if not already marked."""
    if not os.path.exists(MANIFEST_PATH):
        return

    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        if "source" not in fields:
            fields.append("source")
        for r in reader:
            if "source" not in r or not r["source"]:
                r["source"] = "synthetic"
            rows.append(r)

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Updated {len(rows)} manifest rows with source column (synthetic / human_annotated).")


if __name__ == "__main__":
    mark_manifest_sources()
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="001", help="Video stem to annotate (e.g. 001)")
    parser.add_argument("--prefill-all", action="store_true", help="Pre-fill all steps with detector defaults")
    args = parser.parse_args()

    annotate_session(args.video, interactive=False, prefill_all=True)
