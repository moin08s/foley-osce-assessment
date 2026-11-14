import pymupdf
import re
import json

pdf_path = r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf"
doc = pymupdf.open(pdf_path)

with open(r"C:\Medical\dataset\cohort_ground_truth.json") as f:
    cohort = json.load(f)

print(f"Total pages: {len(doc)}")
print("=" * 80)
print(f"{'Page':5s} | {'Top Text / Candidate ID':30s} | {'Cohort Stem':12s} | {'Cohort Grade':12s}")
print("-" * 80)

for idx, page in enumerate(doc):
    txt = page.get_text()
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    top_header = " // ".join(lines[:6])
    
    # Check matching stem in cohort
    # Stems in cohort are "001", "002", "003", ... "025" (note: 006 is missing, so 24 stems!)
    cohort_stems = sorted(list(cohort.keys()))
    assigned_stem = cohort_stems[idx] if idx < len(cohort_stems) else "N/A"
    assigned_grade = cohort.get(assigned_stem, {}).get("faculty_grade", "N/A")
    assigned_name = cohort.get(assigned_stem, {}).get("candidate_name", "N/A")
    
    print(f"P.{idx+1:02d}  | {top_header[:35]:35s} | {assigned_stem:5s} ({assigned_name[:10]}): {assigned_grade:5s}")
