import pymupdf
import cv2
import numpy as np

doc = pymupdf.open(r"C:\text\assesment.pdf")

for p in range(len(doc)):
    page = doc[p]
    pix = page.get_pixmap(dpi=150)
    img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    
    # Let's crop each option box
    # Novice: y 1390:1445, x 360:480
    # Advanced Beginner: y 1390:1445, x 580:820
    # Needs Remediation: y 1390:1445, x 880:1120
    
    # We want to measure the non-white ink that differs from blank template
    # Convert to grayscale and threshold < 200
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Also check HSV for blue/colored pen
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_pen = cv2.inRange(hsv, (85, 30, 30), (140, 255, 255))
    
    # Dark ink below 100
    dark_ink = (gray < 110).astype(np.uint8) * 255
    
    # Let's see: around the words, how many blue pen pixels or extra strokes exist?
    nov_blue = int(np.sum(blue_pen[1380:1460, 350:500] > 0))
    adv_blue = int(np.sum(blue_pen[1380:1460, 560:840] > 0))
    rem_blue = int(np.sum(blue_pen[1380:1460, 860:1140] > 0))
    
    # Also let's inspect if there's black pen or pencil:
    # We can check ink above/below the baseline y: 1360 to 1400 or 1430 to 1470
    nov_top_dark = int(np.sum(dark_ink[1360:1400, 370:480] > 0))
    adv_top_dark = int(np.sum(dark_ink[1360:1400, 600:800] > 0))
    rem_top_dark = int(np.sum(dark_ink[1360:1400, 900:1100] > 0))
    
    print(f"P{p+1:02d} | Blue: Nov={nov_blue:3d}, Adv={adv_blue:3d}, Rem={rem_blue:3d} | TopDark: Nov={nov_top_dark:3d}, Adv={adv_top_dark:3d}, Rem={rem_top_dark:3d}")
