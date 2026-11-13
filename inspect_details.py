import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open(r"C:\Medical\ocr_assesment_pages.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for page_idx_str, page_info in data.items():
    p_num = int(page_idx_str) + 1
    lines = page_info["lines"]
    
    # Search for resident ID
    res_id = None
    for l in lines[:10]:
        if "resident" in l.lower() or "r-" in l.lower():
            res_id = l
            break
            
    # Search for student name
    name = None
    for i, l in enumerate(lines[:12]):
        if "student" in l.lower():
            if ":" in l and len(l.split(":", 1)[1].strip()) > 1:
                name = l.split(":", 1)[1].strip()
            elif i + 1 < len(lines):
                name = lines[i+1]
            break
            
    # Search for adjudication at bottom
    adj = []
    for l in lines[-15:]:
        if any(w in l.lower() for w in ["advanced", "beginner", "novice", "remediation", "prompt", "result", "pass"]):
            adj.append(l)
            
    print(f"Page {p_num:02d} | ID: {res_id} | Name: {name} | Adj: {' / '.join(adj)}")
