import pymupdf
import json

pdf_path = r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf"
doc = pymupdf.open(pdf_path)

for p_num in range(19, 25):
    page = doc[p_num - 1]
    txt = page.get_text()
    print(f"\n==================== PDF PAGE {p_num} ====================")
    for l in txt.split("\n"):
        if l.strip():
            print("  ", l.strip())
