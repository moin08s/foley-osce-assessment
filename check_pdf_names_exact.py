import pymupdf

doc = pymupdf.open(r"C:\New folder (2)\103NZ_30\OSCE Assessment sheet on 04112026 PG Students (1).pdf")

print("=" * 80)
print(f"{'Page':5s} | {'Top Text (Roll / ID)':22s} | {'Student Name in PDF'}")
print("-" * 80)

for p_num in range(1, len(doc) + 1):
    page = doc[p_num - 1]
    lines = [l.strip() for l in page.get_text().split("\n") if l.strip()]
    top_t = lines[0] if lines else "N/A"
    
    # Search for student name
    sname = "Not found"
    for i, l in enumerate(lines):
        if "student" in l.lower() and "name" in l.lower():
            if ":" in l and len(l.split(":")[1].strip()) > 1:
                sname = l.split(":")[1].strip()
            elif i + 1 < len(lines):
                sname = lines[i+1].strip()
            break
            
    print(f"P.{p_num:02d}  | {top_t[:22]:22s} | {sname}")
