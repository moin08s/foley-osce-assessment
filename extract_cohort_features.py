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
out_dir = r"C:\Medical\dataset\features"
os.makedirs(out_dir, exist_ok=True)

def get_step_sequence(stem: str, dur_sec: float) -> list:
    """Load or construct step sequence at 0.5s resolution."""
    batch2_file = os.path.join(gt_dir, f"batch2_{stem}.txt")
    if os.path.exists(batch2_file):
        with open(batch2_file) as f:
            return [int(line.strip()) for line in f if line.strip()]
            
    if stem == "021":
        f21 = os.path.join(gt_dir, "feedback_Candidate_021.txt")
        if os.path.exists(f21):
            with open(f21) as f:
                raw = [int(line.strip()) for line in f if line.strip()]
                # Resample to match duration at 0.5s
                target_len = int(round(dur_sec * 2.0))
                indices = np.linspace(0, len(raw) - 1, target_len).astype(int)
                return [raw[i] for i in indices]

    # Proportional procedural timeline for 13 steps
    # Typical relative proportions of Foley steps:
    weights = [
        0.05, # 1: Intro (5%)
        0.05, # 2: Consent (5%)
        0.07, # 3: Instruments (7%)
        0.09, # 4: Gloves (9%)
        0.07, # 5: Privacy (7%)
        0.13, # 6: Painting (13%)
        0.10, # 7: Lignocaine (10%)
        0.16, # 8: Insertion (16%)
        0.08, # 9: Urobag (8%)
        0.08, # 10: Balloon (8%)
        0.04, # 11: Traction (4%)
        0.04, # 12: Prepuce (4%)
        0.04  # 13: Dressing (4%)
    ]
    weights = np.array(weights) / sum(weights)
    
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

print(f"Starting feature extraction for {len(cohort)} candidates...")
total_start = time.time()

extracted_candidates = []
total_windows_all = 0

for stem, cand in cohort.items():
    out_file = os.path.join(out_dir, f"{stem}_feat.npz")
    if os.path.exists(out_file):
        data = np.load(out_file)
        print(f"[{stem}] Loaded cached features: X={data['X'].shape}, y={len(data['y'])} windows")
        total_windows_all += len(data["y"])
        extracted_candidates.append(stem)
        continue
        
    t0 = time.time()
    vid_path = cand["video_file"]
    aud_path = cand["audio_file"]
    
    if not os.path.exists(vid_path) or not os.path.exists(aud_path):
        print(f"[{stem}] Warning: missing video or audio file, skipping.")
        continue
        
    try:
        audio, sr = load_wav_audio(aud_path, target_sr=16000)
        frames, fps = load_video_frames(vid_path, target_fps=2, frame_size=(64, 64))
        
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
        print(f"[{stem}] Extracted {len(y)} windows in {elapsed:.2f}s | Labels: {sorted(list(set(y)))}")
        total_windows_all += len(y)
        extracted_candidates.append(stem)
    except Exception as e:
        print(f"[{stem}] ERROR during extraction: {e}")

print(f"\nCompleted cohort feature extraction: {len(extracted_candidates)} candidates, {total_windows_all} total windows in {time.time()-total_start:.1f}s")
