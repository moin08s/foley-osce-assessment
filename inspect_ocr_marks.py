import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open(r"C:\Medical\ocr_assesment_pages.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for p in range(len(data)):
    items = data[str(p)]["raw_items"]
    table_marks = []
    bottom_marks = []
    
    for it in items:
        x = it["box"][0][0]
        y = it["box"][0][1]
        text = it["text"]
        
        # Table mark column
        if x > 850 and 400 <= y <= 1250:
            table_marks.append((int(y), int(x), text))
        # Adjudication area
        elif y > 1350:
            bottom_marks.append((int(y), int(x), text))
            
    print(f"\n=== PAGE {p+1:02d} ===")
    print(f"Table marks ({len(table_marks)}):")
    for y, x, txt in table_marks:
        print(f"  y={y:4d}, x={x:4d}: {txt}")
    print(f"Bottom items ({len(bottom_marks)}):")
    for y, x, txt in bottom_marks:
        print(f"  y={y:4d}, x={x:4d}: {txt}")
