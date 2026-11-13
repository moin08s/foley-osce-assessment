import pymupdf
import cv2
import numpy as np
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

doc = pymupdf.open(r"C:\text\assesment.pdf")

# Step Y-ranges at 150 DPI
# From inspection:
# Step 1: y ~ 410-470
# Step 2: y ~ 470-530
# Step 3: y ~ 530-590
# Step 4: y ~ 590-650
# Step 5: y ~ 650-710
# Step 6: y ~ 710-770
# Step 7: y ~ 770-830
# Step 8: y ~ 830-890
# Step 9: y ~ 890-970
# Step 10: y ~ 970-1040
# Step 11: y ~ 1040-1100
# Step 12: y ~ 1100-1160
# Step 13: y ~ 1160-1230

step_y_bands = [
    (1, 410, 470, "Introduce yourself and Explain Procedure"),
    (2, 470, 530, "Informed valid consent"),
    (3, 530, 590, "Make sure all instruments are ready"),
    (4, 590, 650, "Wash Hands and Apply Sanitizer, Wear Sterile gloves"),
    (5, 650, 710, "Checks for privacy of the patient, puts patient in proper position and exposure"),
    (6, 710, 770, "Proper painting and draping"),
    (7, 770, 830, "Application of lignocaine jelly and waits for action"),
    (8, 830, 890, "Takes foley's out in sterile technique and insertion till Y junction"),
    (9, 890, 970, "Attach urobag and confirmation of presence of urine in Urobag, tube"),
    (10, 970, 1040, "Inflation of bulb with distilled water"),
    (11, 1040, 1100, "Pulls out foley's till resistance"),
    (12, 1100, 1160, "Repositioning of prepuce"),
    (13, 1160, 1230, "Catheter tip dressing"),
]

page_summaries = []

for p in range(len(doc)):
    page = doc[p]
    pix = page.get_pixmap(dpi=150)
    img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (90, 35, 35), (135, 255, 255))
    
    # 1. Resident ID in top right (y: 40-120, x: 800-1200)
    top_blue = blue_mask[40:120, 800:1200]
    
    # 2. Check each step
    steps_status = {}
    for sid, y1, y2, sname in step_y_bands:
        prompt_col = blue_mask[y1:y2, 910:980]
        complete_col = blue_mask[y1:y2, 980:1100]
        
        p_count = int(np.sum(prompt_col > 0))
        c_count = int(np.sum(complete_col > 0))
        
        if p_count > 50 and p_count > c_count * 0.8:
            status = "P-1 (Prompt)"
        elif c_count > 30:
            status = "PASS (✔)"
        else:
            status = "MISSED (✘)"
            
        steps_status[sid] = {
            "status": status,
            "prompt_pixels": p_count,
            "complete_pixels": c_count
        }
        
    # 3. Overall Adjudication (y: 1380 to 1480)
    novice_box = blue_mask[1380:1480, 350:520]
    adv_box = blue_mask[1380:1480, 580:860]
    remed_box = blue_mask[1380:1480, 880:1150]
    
    n_count = int(np.sum(novice_box > 0))
    a_count = int(np.sum(adv_box > 0))
    r_count = int(np.sum(remed_box > 0))
    
    if r_count > 80 and r_count > a_count and r_count > n_count:
        final_adj = "R: Needs Remediation"
    elif n_count > 80 and n_count > a_count:
        final_adj = "P-1: Novice (Pass after Prompt)"
    elif a_count > 80:
        final_adj = "P: Advance Beginner (Pass)"
    else:
        # Check missed steps
        missed = sum(1 for s in steps_status.values() if "MISSED" in s["status"])
        if missed >= 2:
            final_adj = "R: Needs Remediation"
        elif any("P-1" in s["status"] for s in steps_status.values()):
            final_adj = "P-1: Novice (Pass after Prompt)"
        else:
            final_adj = "P: Advance Beginner (Pass)"
            
    summary = {
        "page": p + 1,
        "adjudication": final_adj,
        "ink_counts": {"novice": n_count, "adv": a_count, "remed": r_count},
        "missed_steps": [sid for sid, s in steps_status.items() if "MISSED" in s["status"]],
        "p1_steps": [sid for sid, s in steps_status.items() if "P-1" in s["status"]],
    }
    page_summaries.append(summary)
    print(f"Page {p+1:02d}: {final_adj:30s} | Missed: {summary['missed_steps']} | P-1: {summary['p1_steps']} | ink: (N:{n_count}, A:{a_count}, R:{r_count})")
