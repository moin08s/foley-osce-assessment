import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open(r"C:\Medical\ocr_assesment_pages.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for p in range(min(3, len(data))):
    print(f"\n--- PAGE {p+1} ---")
    items = data[str(p)]["raw_items"]
    # Sort by y, then x
    sorted_items = sorted(items, key=lambda x: (x["box"][0][1], x["box"][0][0]))
    for it in sorted_items:
        y = int(it["box"][0][1])
        x = int(it["box"][0][0])
        w = int(it["box"][1][0] - it["box"][0][0])
        h = int(it["box"][2][1] - it["box"][1][1])
        print(f"y={y:4d}, x={x:4d}, w={w:3d}, h={h:3d} | {it['text']}")
