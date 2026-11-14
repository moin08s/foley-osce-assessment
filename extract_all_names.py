import pymupdf
import re

pdf_path = r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf"
doc = pymupdf.open(pdf_path)

for p_num in range(1, 25):
    page = doc[p_num - 1]
    blocks = page.get_text("blocks")
    
    # Sort blocks by y
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    
    # Extract top 5 text blocks
    texts = [b[4].strip() for b in blocks if b[4].strip()]
    
    # Find student name specifically
    sname = "N/A"
    for i, t in enumerate(texts):
        if "student" in t.lower() and "name" in t.lower():
            # If the block itself has the name after ':'
            if ":" in t and len(t.split(":")[1].strip()) > 1:
                sname = t.split(":")[1].strip()
            elif i + 1 < len(texts):
                sname = texts[i+1].replace("\n", " ").strip()
            break
            
    # Find top right or top left roll code
    roll_code = "N/A"
    for t in texts[:8]:
        clean = " ".join(t.split())
        if any(c.isdigit() for c in clean) and len(clean) <= 15:
            roll_code = clean
            break
            
    print(f"Page {p_num:02d} | Roll: {roll_code:15s} | Name: {sname[:25]:25s}")
