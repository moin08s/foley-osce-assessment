"""Extract 128x128, 4 fps multimodal features for all 24 cohort videos (Task 4)."""

import os
import sys
import json
import time
import numpy as np

from src.data_loader import load_video_frames, load_wav_audio, slice_windows
from src.features import extract_window_features

with open(r"C:\Medical\dataset\cohort_ground_truth.json") as f:
    cohort = json.load(f)

gt_dir = r"C:\Medical\catheter_ai_project\dataset\ground_truth"
out_dir = r"C:\Medical\dataset\features_128px"
os.makedirs(out_dir, exist_ok=True)

weights = np.array([0.05, 0.05, 0.07, 0.09, 0.07, 0.13, 0.10, 0.16, 0.08, 0.08, 0.04, 0.04, 0.04])
weights = weights / np.sum(weights)

def get_step_sequence(stem: str, dur_sec: float) -> list:
    batch2_file = os.path.join(gt_dir, f"batch2_{stem}.txt")
    if os.path.exists(batch2_file):
        with open(batch2_file) as f:
            return [int(line.strip()) for line in f if line.strip()]
            
    n_steps = int(round(dur_sec * 2.0))
    seq = np.zeros(n_steps, dtype=int)
    cand_info = cohort.get(stem, {})
    missing = cand_info.get("missing_steps", [])
    cur_idx = 0
    for sid in range(1, 14):
        dur_steps = int(round(weights[sid - 1] * n_steps))
        end_idx = min(cur_idx + dur_steps, n_steps) if sid < 13 else n_steps
        step_val = 0 if sid in missing else sid
        seq[cur_idx:end_idx] = step_val
        cur_idx = end_idx
    return list(seq)

print("Starting 128x128, 4 fps feature extraction for 24 candidate videos...")
t_start = time.time()

for stem, cand in cohort.items():
    out_file = os.path.join(out_dir, f"{stem}_feat.npz")
    if os.path.exists(out_file):
        data = np.load(out_file)
        print(f"[{stem}] Already extracted: X={data['X'].shape}, y={len(data['y'])} windows")
        continue

    t0 = time.time()
    vid_path = cand["video_file"]
    aud_path = cand["audio_file"]

    if not os.path.exists(vid_path) or not os.path.exists(aud_path):
        print(f"[{stem}] Missing video or audio file, skipping.")
        continue

    audio, sr = load_wav_audio(aud_path, target_sr=16000)
    # Extract 128x128 frames at 4 fps
    frames, fps = load_video_frames(vid_path, target_fps=4, frame_size=(128, 128))
    dur_sec = len(audio) / sr

    step_seq = get_step_sequence(stem, dur_sec)
    windows = slice_windows(audio, sr, frames, fps, window_sec=4.0, hop_sec=2.0)

    X_list = []
    y_list = []
    for w in windows:
        feat = extract_window_features(w["video"], w["audio"], sr)
        mid_sec = (w["start_sec"] + w["end_sec"]) / 2.0
        step_idx = int(mid_sec / 0.5)
        lbl = step_seq[min(step_idx, len(step_seq) - 1)]
        X_list.append(feat)
        y_list.append(lbl)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.savez_compressed(out_file, X=X, y=y)
    elapsed = time.time() - t0
    print(f"[{stem}] Extracted 128x128 4fps: {len(y)} windows in {elapsed:.1f}s | X={X.shape}")

print(f"\nCompleted 128x128 4fps feature extraction in {time.time()-t_start:.1f}s!")
