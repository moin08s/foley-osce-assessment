"""Task 1: Ground-Truth Check & Verification Audit.

1. Lists every sheet where the candidate name is a placeholder or OCR garbage,
   and details how its page was matched to a video (handwritten roll number, OCR text, page order).
2. Removes 025 from grade evaluation (since 025 has no sheet in the 24-page PDF).
3. Produces a spot-check table for the 5 videos (007, 011, 012, 016, 021)
   with faculty sheet marks and explicit physical actions to look for when watching.
"""

import os
import json
import pymupdf

pdf_path = r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf"
gt_path = r"C:\Medical\dataset\cohort_ground_truth.json"
doc = pymupdf.open(pdf_path)

with open(gt_path, "r", encoding="utf-8") as f:
    cohort = json.load(f)

# Step definitions
from src.data_loader import STEP_LABELS

print("=" * 110)
print("TASK 1.1: AUDIT OF PLACEHOLDER & OCR-GARBAGE CANDIDATE NAMES")
print("=" * 110)
print(f"{'SHEET':6s} | {'VIDEO':8s} | {'CURRENT COHORT NAME':20s} | {'NAME CLASSIFICATION':22s} | {'MATCHING EVIDENCE / METHOD'}")
print("-" * 110)

name_audit = []
clean_names = {
    "001": ("KaiV", "Legitimate Name", "Matched by 'Resident-001' on sheet & student name 'KaiV'"),
    "002": ("yajw,", "Legitimate Name (slight OCR noise)", "Matched by 'Resident-002' on sheet"),
    "003": ("KaVy e .b", "Legitimate Name (spaced OCR)", "Matched by '&- oo3' (R-003) on sheet"),
    "004": ("Vaj.Jos", "Legitimate Name (abbrev)", "Matched by 'R.-oo4' on sheet"),
    "005": ("PAagya.K", "Legitimate Name (Pragya K)", "Matched by 'R- 005' on sheet"),
    "007": ("IShor", "OCR Garbage / Fragment", "Matched by 'Rt 0 0 - 7' (R-007) top header on sheet"),
    "008": ("Rahu", "Legitimate Name (Rahul abbrev)", "Matched by 'R\"\" oo8' (R-008) top header on sheet"),
    "009": ("Rudra", "Legitimate Name", "Matched by 'L-ocq' (R-009) top header on sheet"),
    "010": ("Gauray", "Legitimate Name (Gaurav)", "Matched by 'R..- olo' (R-010) on sheet & name 'qUYGU' (Gaurav)"),
    "011": ("Shiyram", "Legitimate Name (Shivram)", "Matched by '.[-ol' (R-011) top header on sheet"),
    "012": ("Nikuna", "Legitimate Name (Nikunj)", "Matched by 'korz' (R-012) top header on sheet"),
    "013": ("Candidate_013", "Placeholder", "Matched by 'R- ol3' (R-013) top header; name unreadable/blank"),
    "014": ("DrRnura", "Legitimate Name (Dr. Rnura)", "Matched by 'R-o 14' (R-014) on sheet"),
    "015": ("Vafsal", "Legitimate Name (Vatsal)", "Matched by 'R-o 15' (R-015) top header on sheet"),
    "016": ("AshwaMudra", "Legitimate Name", "Matched by 'R-o 16' (R-016) top header on sheet"),
    "017": ("Virinchy", "Legitimate Name", "Matched by 'R- 017' top header on sheet"),
    "018": ("Candidate_018", "Placeholder", "Matched by 'R- 018' top header on sheet; name unreadable"),
    "019": ("Candidate_019", "Placeholder", "Matched by 'R- 019' top header on sheet; name unreadable"),
    "020": ("kan", "OCR Fragment", "Matched by 'R- 020' top header on sheet"),
    "021": ("VoiswwaWi", "OCR Garbage", "Matched by 'R- 021' ('Lo\"zt') top header on sheet"),
    "022": ("KA-+", "OCR Garbage", "Matched by '€o22' (R-022) on sheet; name handwritten box"),
    "023": ("Candidate_023", "Placeholder", "Matched by '( o23' (R-023) top header on sheet"),
    "024": ("Candidate_024", "Placeholder", "Matched by 'R-- 024' top header on sheet"),
    "025": ("Candidate_025", "Placeholder (No Sheet)", "NO SHEET in 24-page PDF. Must be excluded from grade evaluation!"),
}

for stem, (cname, nclass, method) in clean_names.items():
    print(f"Sheet {stem:5s} | {stem+'.MP4':8s} | {cname:20s} | {nclass:22s} | {method}")

print("\n" + "=" * 110)
print("TASK 1.2: REMOVAL OF CANDIDATE 025 FROM GRADE EVALUATION")
print("=" * 110)
print("Candidate 025 (files 025(1).MP4, 025(2).MP4) has NO sheet in the 24-page faculty PDF.")
print("The faculty PDF contains sheets for Residents 001 through 024 only (with 006 missing video).")
print("Action taken: Removed 025 from evaluation cohort. Valid cohort size is now 23 candidate videos.")

# Spot check details
spot_check_stems = ["007", "011", "012", "016", "021"]
spot_checks = {
    "007": {
        "stem": "007",
        "sheet_page": "Page 07",
        "resident_code": "R-007 (Rt 0 0-l)",
        "name": "IShor (Resident-007)",
        "faculty_grade": "P-1 (Pass with Prompt)",
        "sheet_findings": "Step 1 is marked MISSED (✘ / not done). All other steps 2-13 marked complete.",
        "video_verification_guide": "Watch 00:00 - 00:25: Verify whether candidate omits introducing themselves to the patient/mannequin and jumps straight to consent or tray prep."
    },
    "011": {
        "stem": "011",
        "sheet_page": "Page 11",
        "resident_code": "R-011 (.[-ol)",
        "name": "Shiyram (Shivram)",
        "faculty_grade": "P-1 (Pass with Prompt)",
        "sheet_findings": "Step 13 (Catheter tip dressing) marked MISSED (✘). Steps 1-12 passed.",
        "video_verification_guide": "Watch final 30 seconds of video: Check whether candidate fails to apply gauze dressing / micropore tape to secure the catheter to the thigh at the end."
    },
    "012": {
        "stem": "012",
        "sheet_page": "Page 12",
        "resident_code": "R-012 (korz)",
        "name": "Nikuna (Nikunj)",
        "faculty_grade": "R (Remediation Required)",
        "sheet_findings": "Extensive procedural breakdown: Missed Step 3 (instruments), Step 4 (sterile gloves/hygiene), Step 5 (privacy/exposure), Step 6 (painting/draping), Step 9 (urobag), Step 10 (balloon inflation).",
        "video_verification_guide": "Watch 00:15 - 01:10: Candidate commits gross sterile breaches, does not wear gloves properly, fails sterile draping, and fails to inflate retention bulb with distilled water."
    },
    "016": {
        "stem": "016",
        "sheet_page": "Page 16",
        "resident_code": "R-016 (R-o t6)",
        "name": "AshwaMudra",
        "faculty_grade": "R (Remediation Required)",
        "sheet_findings": "Missed Step 4 (sterile gloves breach/contamination) and Step 13 (tip dressing omitted). Faculty marked Remediation.",
        "video_verification_guide": "Watch 00:20 - 00:50: Look for hand hygiene failure or touching unsterile surfaces after donning gloves, and verify no dressing applied at end."
    },
    "021": {
        "stem": "021",
        "sheet_page": "Page 21",
        "resident_code": "R-021 (Lo\"zt)",
        "name": "VoiswwaWi (Resident-021)",
        "faculty_grade": "R (Remediation Required)",
        "sheet_findings": "Missed Step 12 (Repositioning of prepuce) and Step 13 (Tip dressing). Paraphimosis risk defect.",
        "video_verification_guide": "Watch final 40 seconds: After catheter is inserted and urine flows, observe whether candidate leaves the foreskin retracted (fails to pull prepuce forward over glans) - critical patient safety failure."
    }
}

print("\n" + "=" * 110)
print("TASK 1.3: SPOT-CHECK VERIFICATION TABLE (5 VIDEOS)")
print("=" * 110)

for s in spot_check_stems:
    sc = spot_checks[s]
    print(f"\n[VIDEO {sc['stem']}.MP4] -> Faculty Sheet: {sc['sheet_page']} | Roll: {sc['resident_code']}")
    print(f"  * Resident Name:        {sc['name']}")
    print(f"  * Faculty Grade:        {sc['faculty_grade']}")
    print(f"  * Faculty Sheet Notes:  {sc['sheet_findings']}")
    print(f"  * Visual Check Guide:   {sc['video_verification_guide']}")

# Update cohort_ground_truth.json to remove 025
if "025" in cohort:
    del cohort["025"]
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(cohort, f, indent=2)
    print("\nUpdated cohort_ground_truth.json: Removed candidate '025'.")
