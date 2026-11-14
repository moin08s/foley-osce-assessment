import pymupdf
import re

pdf_path = r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf"
doc = pymupdf.open(pdf_path)

print(f"{'Page':5s} | {'Raw Text Near Top/Header':55s}")
print("-" * 70)

for idx, page in enumerate(doc):
    txt = page.get_text()
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    
    # Extract any lines with numbers or 'R-'
    cand_lines = [l for l in lines[:10] if any(c.isdigit() for c in l) or 'r' in l.lower()]
    sample = " // ".join(cand_lines[:3])
    print(f"P.{idx+1:02d}  | {sample:55s}")
