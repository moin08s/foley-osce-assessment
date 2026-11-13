import pymupdf
import cv2
import numpy as np

doc = pymupdf.open(r"C:\text\assesment.pdf")

for p in range(len(doc)):
    page = doc[p]
    pix = page.get_pixmap(dpi=150)
    img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    
    # Bottom area (y: 1350 to 1500, x: 150 to 1150)
    # Check ink density around Novice (x: 350-500), Advanced Beginner (x: 600-850), Needs Remediation (x: 900-1150)
    # in y: 1380 to 1480
    bottom_crop = img[1380:1480, 150:1150]
    
    # Save bottom crop for inspection
    # cv2.imwrite(f"C:\\Medical\\bottom_p{p+1}.png", bottom_crop)
    
    # Count blue/dark pen pixels or ticks in the columns
    # Right table columns: Complete (x: 950-1100), One Prompt (x: 880-970)
    print(f"Page {p+1:02d}: shape {img.shape}")
