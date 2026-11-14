import pymupdf
import re

pdf_path = r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf"
doc = pymupdf.open(pdf_path)

print("=" * 100)
print(f"{'Page':5s} | {'Top Number':14s} | {'Student Name':20s} | {'Adjudged / Rating text'}")
print("-" * 100)

for idx, page in enumerate(doc):
    txt = page.get_text()
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    
    # Top candidate code
    top_code = lines[0] if lines else ""
    
    # Find student name
    sname = "Unknown"
    for i, l in enumerate(lines):
        if "student" in l.lower() and "name" in l.lower():
            if i + 1 < len(lines):
                sname = lines[i+1]
            break
            
    # Find bottom adjudication
    adj_lines = []
    for i, l in enumerate(lines):
        if any(w in l.lower() for w in ["adjudged", "competent", "advance", "beginner", "novice", "remediation", "completed as"]):
            adj_lines.append(l)
            if i + 1 < len(lines):
                adj_lines.append(lines[i+1])
                
    adj_str = " // ".join(adj_lines[:4])
    print(f"P.{idx+1:02d}  | {top_code[:12]:12s} | {sname[:20]:20s} | {adj_str}")
