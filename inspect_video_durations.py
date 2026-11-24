"""Inspect video durations and properties for all 23 cohort videos."""

import os
import cv2
import json

GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"

with open(GT_PATH) as f:
    cohort = json.load(f)

print(f"{'STEM':<6} | {'NAME':<16} | {'FILE':<12} | {'DURATION (s)':<14} | {'FPS':<6} | {'FRAMES':<8} | {'WIDTHxHEIGHT'}")
print("-" * 80)

for stem in sorted(cohort.keys()):
    cand = cohort[stem]
    vpath = cand["video_file"]
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = count / fps if fps > 0 else 0
    cap.release()
    print(f"{stem:<6} | {cand['candidate_name']:<16} | {os.path.basename(vpath):<12} | {dur:<14.1f} | {fps:<6.1f} | {int(count):<8} | {w}x{h}")
