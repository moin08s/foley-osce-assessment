import pymupdf
import cv2
import numpy as np

doc = pymupdf.open(r"C:\text\assesment.pdf")

for p in range(min(5, len(doc))):
    page = doc[p]
    pix = page.get_pixmap(dpi=150)
    img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    
    # Check bottom adjudication area: y: 1380 to 1480, x: 300 to 1100
    # Novice box roughly x: 350-500
    # Advanced Beginner box roughly x: 580-850
    # Needs Remediation box roughly x: 880-1150
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Check blue ink mask (H: 90-135, S: 50-255, V: 30-240)
    blue_mask = cv2.inRange(hsv, (90, 40, 40), (135, 255, 255))
    
    # Check dark ink (black pen / pencil) below grayscale threshold < 80
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < 90).astype(np.uint8) * 255
    
    blue_count_total = int(np.sum(blue_mask > 0))
    print(f"Page {p+1:02d}: blue ink pixels = {blue_count_total}")
    
    # Save a small composite of the adjudication area
    adj_area = img[1380:1480, 300:1150]
    cv2.imwrite(f"C:\\Medical\\adj_p{p+1}.png", adj_area)
