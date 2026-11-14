import pymupdf
import re

pdf_path = r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf"
doc = pymupdf.open(pdf_path)

print(f"{'Page':5s} | {'Top Number / Roll':20s} | {'Student Name Block':25s} | {'Bottom Adjudication Text':35s}")
print("-" * 95)

for p_idx in range(len(doc)):
    page = doc[p_idx]
    blocks = page.get_text("blocks")
    
    top_texts = []
    name_texts = []
    bottom_texts = []
    step_marks = {}
    
    for b in blocks:
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
        if not text:
            continue
            
        # Top area (resident number / roll)
        if y1 < 130:
            # Check if has numbers or R-
            clean_t = " ".join(text.split())
            if any(c.isdigit() for c in clean_t) or "r" in clean_t.lower() or "oo" in clean_t.lower() or "ol" in clean_t.lower():
                top_texts.append(clean_t)
                
        # Student Name area (y between 100 and 190)
        if 100 <= y0 <= 190:
            clean_t = " ".join(text.split())
            if "student" in clean_t.lower() or "nam" in clean_t.lower():
                name_texts.append(clean_t)
            elif not any(h in clean_t for h in ["D Y Patil", "OSCE", "Skill", "Foley"]):
                name_texts.append(clean_t)
                
        # Bottom adjudication area (y > 550)
        if y0 >= 550:
            clean_t = " ".join(text.split())
            if any(w in clean_t.lower() for w in ["adjudged", "competent", "beginner", "novice", "remediation", "p-1", "p-", "pass", "fail", "needs"]):
                bottom_texts.append(clean_t)
                
    top_str = " // ".join(top_texts)[:20]
    name_str = " // ".join(name_texts)[:25]
    bot_str = " // ".join(bottom_texts)[:35]
    
    print(f"P.{p_idx+1:02d}  | {top_str:20s} | {name_str:25s} | {bot_str:35s}")
