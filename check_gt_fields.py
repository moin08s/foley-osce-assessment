import json

with open(r"C:\Medical\dataset\cohort_ground_truth.json") as f:
    d = json.load(f)

for k in ["007", "011", "012", "016", "021"]:
    print(k, d[k])
