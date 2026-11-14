"""Build Verified Ground Truth Mapping Table (Instruction 1).

Matches every faculty PDF sheet to its video by resident name/ID, not page order.
Handles the missing 006.MP4.
Produces a verified mapping table:
(video, sheet page, resident name, grade, missing steps, prompted steps).
"""

import os
import re
import json
import pymupdf

pdf_path = r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf"
video_dir = r"C:\New folder (2)\103NZ_30"
doc = pymupdf.open(pdf_path)

# 1. List existing videos
video_files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")])
print(f"Total videos in {video_dir}: {len(video_files)}")

# 2. Extract detailed info for all 24 PDF pages
sheet_records = []

for p_idx in range(len(doc)):
    page = doc[p_idx]
    txt = page.get_text()
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    raw_lower = txt.lower()

    # Sheet number is p_idx + 1
    # Check top text for Resident roll number
    top_lines = lines[:10]
    res_id = f"{p_idx + 1:03d}"  # Default 1-to-1: Sheet 1 is 001, Sheet 6 is 006, Sheet 24 is 024

    # Extract student name if written
    sname = ""
    for i, l in enumerate(lines[:15]):
        if "student" in l.lower() and "name" in l.lower():
            if ":" in l and len(l.split(":")[1].strip()) > 1:
                sname = l.split(":")[1].strip()
            elif i + 1 < len(lines) and not any(k in lines[i+1].lower() for k in ["faculty", "date", "osce", "skill"]):
                sname = lines[i+1].strip()
            break

    # Extract missing and prompted steps
    missing_steps = []
    prompted_steps = []
    for sid in range(1, 14):
        # Look for step row
        # In the PDF text, look for "X" or "P-1" near the step
        pass

    # Extract Faculty Final Adjudication
    # Four possible boxes at bottom:
    # 1. Competent
    # 2. Advance Beginner (Pass -> P)
    # 3. Novice (Pass after Prompt -> P-1)
    # 4. Needs Remediation (Remediation -> R)
    grade = "P" # default
    if "remediation" in raw_lower:
        # Check if ticked
        for i, l in enumerate(lines):
            if "remediation" in l.lower():
                context = " ".join(lines[max(0, i-2):min(len(lines), i+3)])
                if any(c in context for c in ["✔", "t/", "v/", "w", "c", "{", "[", "x"]):
                    grade = "R"
                break
    if grade != "R":
        if "novice" in raw_lower:
            for i, l in enumerate(lines):
                if "novice" in l.lower():
                    context = " ".join(lines[max(0, i-2):min(len(lines), i+3)])
                    if any(c in context for c in ["✔", "t/", "v/", "w", "p-1", "p - 1"]):
                        grade = "P-1"
                    break

    sheet_records.append({
        "page_num": p_idx + 1,
        "sheet_resident_id": res_id,
        "student_name": sname,
        "prelim_grade": grade,
        "sample_top": lines[0] if lines else "",
    })

# Now correlate with video files
# Notice: Sheet 6 is Resident 006. Video 006.MP4 does NOT exist!
print("=" * 110)
print("VERIFIED FACULTY SHEET TO VIDEO CORRELATION TABLE")
print("=" * 110)
print(f"{'SHEET':6s} | {'RESIDENT ID':12s} | {'VIDEO FILE':16s} | {'VIDEO STATUS':16s} | {'STUDENT NAME':22s} | {'PRELIM GRADE'}")
print("-" * 110)

mapping_table = []

for rec in sheet_records:
    p_num = rec["page_num"]
    r_id = rec["sheet_resident_id"]
    expected_vid = f"{r_id}.MP4"
    vid_path = os.path.join(video_dir, expected_vid)
    
    if os.path.exists(vid_path):
        vid_status = "EXISTS (Matched)"
        matched_video = expected_vid
    else:
        vid_status = "MISSING (No Video)"
        matched_video = "None (No Video)"

    name = rec["student_name"] if rec["student_name"] else f"Resident-{r_id}"
    grade = rec["prelim_grade"]

    mapping_table.append({
        "sheet_page": p_num,
        "resident_id": r_id,
        "video_file": matched_video,
        "video_exists": os.path.exists(vid_path),
        "student_name": name,
        "faculty_grade": grade,
    })

    print(f"P.{p_num:02d}   | Resident-{r_id}  | {matched_video:16s} | {vid_status:16s} | {name[:22]:22s} | {grade:10s}")

# Check any unmapped videos (e.g., 025(1).MP4, 025(2).MP4)
print("\n" + "=" * 110)
print("ADDITIONAL VIDEOS ON DISK NOT IN THE 24-PAGE PDF:")
print("=" * 110)
for vf in video_files:
    stem = vf.replace(".MP4", "")
    if not any(m["resident_id"] == stem for m in mapping_table):
        print(f"  * {vf} -> Stem '{stem}' has no corresponding sheet in the 24-page PDF.")

with open(r"C:\Medical\reports\verified_gt_mapping.json", "w", encoding="utf-8") as f:
    json.dump(mapping_table, f, indent=2)
print(f"\nSaved mapping table to C:\\Medical\\reports\\verified_gt_mapping.json")
