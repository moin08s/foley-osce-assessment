import pymupdf
import cv2
import numpy as np

doc = pymupdf.open(r"C:\text\assesment.pdf")

for p in range(3):
    page = doc[p]
    pix = page.get_pixmap(dpi=150)
    img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (90, 35, 35), (135, 255, 255))
    
    # Find all blue contours
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marks = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 20: # significant ink mark
            x, y, w, h = cv2.boundingRect(c)
            marks.append((y, x, w, h, int(area)))
            
    marks.sort(key=lambda m: m[0])
    print(f"\n--- PAGE {p+1} INK MARKS (total {len(marks)}) ---")
    for y, x, w, h, a in marks:
        # print marks in table area (y between 350 and 1250, x > 850)
        if 350 <= y <= 1250 and x > 850:
            print(f"Table mark at y={y:4d}, x={x:4d}, w={w:3d}, h={h:3d}, area={a:4d}")
        # print marks in bottom adjudication area (y > 1350)
        elif y > 1350:
            print(f"Bottom mark at y={y:4d}, x={x:4d}, w={w:3d}, h={h:3d}, area={a:4d}")
