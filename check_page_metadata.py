import pymupdf
import re
import json

pdf_path = r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf"
doc = pymupdf.open(pdf_path)

for idx, page in enumerate(doc):
    txt = page.get_text()
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    
    # Extract student name, faculty name, date, roll no
    header_lines = lines[:15]
    print(f"=== PAGE {idx+1} ===")
    for hl in header_lines:
        print("  ", hl)
    
    # Look for candidate name specifically
    for i, line in enumerate(lines):
        if any(k in line.lower() for k in ["student", "name", "candidate", "roll"]):
            context = " | ".join(lines[max(0, i-1):min(len(lines), i+3)])
            print(f"   [Context]: {context}")
            break
