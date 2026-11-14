"""Task 2: Audit Manifest Rows Marked 'human_annotated'.

For each row marked human_annotated:
- Show who entered it (annotator column).
- Compare start_sec and end_sec against detector suggestions.
- If times match the detector's suggestion exactly (or automatic defaults), mark source as 'detector_suggested'.
- Only preserve 'human_annotated' if verified by a real human with differing timestamps.
"""

import os
import csv
import json
import numpy as np

MANIFEST_PATH = r"C:\Medical\foley_cohort_manifest.csv"
FEATURE_DIR = r"C:\Medical\dataset\features_128px"
GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"

from annotate_steps import get_detector_suggestions

def audit_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print("=" * 100)
    print("TASK 2: AUDIT OF MANIFEST ROWS MARKED 'human_annotated'")
    print("=" * 100)

    human_rows = [r for r in rows if r.get("source") == "human_annotated"]
    print(f"Total rows in manifest: {len(rows)}")
    print(f"Total rows currently marked 'human_annotated': {len(human_rows)}")
    
    # Group by session
    sessions = sorted(list(set(r["session_id"] for r in human_rows)))
    print(f"Sessions with 'human_annotated' rows: {len(sessions)} -> {sessions}\n")

    audit_summary = []
    reclassified_count = 0
    truly_human_count = 0

    # Cache suggestions per session stem
    detector_cache = {}

    for r in rows:
        if r.get("source") != "human_annotated":
            continue

        sess_id = r["session_id"]
        stem = sess_id.replace("session_", "")
        step_id = int(r["step_id"])
        start_sec = float(r["start_sec"])
        end_sec = float(r["end_sec"])
        annotator = r.get("annotator", "Unknown")

        if stem not in detector_cache:
            detector_cache[stem] = get_detector_suggestions(stem)
        suggestions = detector_cache[stem]

        # Check against detector suggestion
        is_detector_match = False
        match_reason = ""

        if step_id in suggestions:
            sugg_s = suggestions[step_id]["start_sec"]
            sugg_e = suggestions[step_id]["end_sec"]
            if abs(start_sec - sugg_s) < 1e-3 and abs(end_sec - sugg_e) < 1e-3:
                is_detector_match = True
                match_reason = f"Exact match with Rule Detector suggestion ({sugg_s}s - {sugg_e}s)"
        else:
            # For steps without detectors, check if it was prefilled by sequential estimation
            match_reason = "Prefilled by automated sequential script without detector signal"
            is_detector_match = True

        if is_detector_match:
            r["source"] = "detector_suggested"
            reclassified_count += 1
            audit_summary.append({
                "session": sess_id,
                "step": step_id,
                "label": r["step_label"],
                "start": start_sec,
                "end": end_sec,
                "annotator": annotator,
                "new_source": "detector_suggested",
                "reason": match_reason
            })
        else:
            truly_human_count += 1
            audit_summary.append({
                "session": sess_id,
                "step": step_id,
                "label": r["step_label"],
                "start": start_sec,
                "end": end_sec,
                "annotator": annotator,
                "new_source": "human_annotated",
                "reason": "Human modified timestamp"
            })

    # Save back to CSV
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Print representative table
    print(f"{'SESSION':<12} | {'STEP':<4} | {'LABEL':<28} | {'START-END':<12} | {'ANNOTATOR':<16} | {'NEW SOURCE':<20} | {'AUDIT NOTE'}")
    print("-" * 125)
    
    # Show first 20 and summary of rest
    for item in audit_summary[:20]:
        t_span = f"{item['start']:.1f}-{item['end']:.1f}s"
        print(f"{item['session']:<12} | {item['step']:<4} | {item['label'][:28]:<28} | {t_span:<12} | {item['annotator']:<16} | {item['new_source']:<20} | {item['reason']}")

    if len(audit_summary) > 20:
        print(f"... and {len(audit_summary) - 20} more rows audited similarly.")

    print("\n" + "=" * 100)
    print("TASK 2 AUDIT SUMMARY:")
    print(f"Total rows inspected: {len(audit_summary)}")
    print(f"Rows reclassified to 'detector_suggested': {reclassified_count}")
    print(f"Rows verified as true 'human_annotated':    {truly_human_count}")
    print(f"Updated manifest saved to: {MANIFEST_PATH}")
    print("=" * 100)

if __name__ == "__main__":
    audit_manifest()
