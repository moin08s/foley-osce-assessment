"""Task 4: Extract exact faculty sheet data for spot-check videos.

Strict requirement:
Remove the "what to look for" explanations that are not in the faculty sheet.
Keep only: video, sheet page, faculty-marked step, and the sheet's own comment text if any.
"""

import json

GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"

def clean_spot_check():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    spot_keys = ["007", "011", "012", "016", "021"]

    print("=" * 100)
    print("TASK 4: VERIFIED SPOT-CHECK TABLE (FACULTY SHEET DATA ONLY)")
    print("=" * 100)
    print(f"{'VIDEO':<10} | {'PAGE':<8} | {'FACULTY-MARKED STEP(S)':<35} | {'SHEET COMMENT / REMARKS TEXT'}")
    print("-" * 100)

    for k in spot_keys:
        cand = cohort.get(k, {})
        page_num = cand.get("pdf_page_index", 0) + 1
        missing = cand.get("missing_steps", [])
        
        if not missing:
            marked_steps = "All steps passed (None missed)"
        else:
            marked_steps = f"Step(s) marked MISSED: {missing}"

        # Sheet comment text
        comment = cand.get("faculty_comments") or cand.get("remarks") or "None"

        print(f"{k + '.MP4':<10} | Page {page_num:<3d} | {marked_steps:<35} | {comment}")

    print("=" * 100)

if __name__ == "__main__":
    clean_spot_check()
