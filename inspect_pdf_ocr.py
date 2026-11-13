import os
import sys
import json
import pymupdf
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pdf_path = r"C:\text\assesment.pdf"
doc = pymupdf.open(pdf_path)
ocr = RapidOCR()

results = {}

for page_idx in range(len(doc)):
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    ocr_res, _ = ocr(img_np)
    lines = [item[1] for item in ocr_res] if ocr_res else []
    
    # Store text and bounding boxes
    page_data = {
        "page_idx": page_idx,
        "lines": lines,
        "raw_items": [
            {"box": item[0], "text": item[1], "score": float(item[2])}
            for item in (ocr_res or [])
        ]
    }
    
    header = " | ".join(lines[:8])
    print(f"Page {page_idx + 1:02d}: {header}")
    results[page_idx] = page_data

with open(r"C:\Medical\ocr_assesment_pages.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nSaved OCR results to C:\\Medical\\ocr_assesment_pages.json")
