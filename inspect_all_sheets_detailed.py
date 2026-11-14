import pymupdf
import re

pdf_path = r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf"
doc = pymupdf.open(pdf_path)

print(f"Total pages in PDF: {len(doc)}")

for p_idx in range(len(doc)):
    page = doc[p_idx]
    txt = page.get_text()
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    
    # Extract candidate code / resident number
    # Let's inspect first 10 lines and look for candidate name, date, roll no
    header_block = lines[:10]
    
    # Look for Student Name line
    sname = "Not explicit"
    for i, l in enumerate(lines):
        if "student" in l.lower() and "name" in l.lower():
            if i + 1 < len(lines):
                sname = lines[i+1]
            break

    # Look for Adjudged / Grade box
    grade = "Unknown"
    # Find ticks near bottom
    for i, l in enumerate(lines):
        if "remediation" in l.lower():
            # Check context
            pass
            
    print(f"--- PAGE {p_idx+1} ---")
    print("  Header lines:", header_block[:5])
    print(f"  Student Name: {sname}")
