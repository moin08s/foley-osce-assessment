"""Test the annotation saving and loading logic."""

import os
import csv
import json

ANNOTATION_CSV = r"C:\Medical\human_annotation_sheet.csv"
GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"

def test_load_cohort():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)
    print(f"Loaded cohort: {len(cohort)} candidates.")
    assert len(cohort) == 23

def test_read_csv():
    assert os.path.exists(ANNOTATION_CSV)
    with open(ANNOTATION_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded CSV: {len(rows)} rows.")
    assert len(rows) == 299

if __name__ == "__main__":
    test_load_cohort()
    test_read_csv()
    print("All tests passed!")
