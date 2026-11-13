"""Extract features and prepare dataset for the 4 videos in C:\\103NZ_30."""

import os
import subprocess
import numpy as np
from src.data_loader import load_video_frames, load_wav_audio, slice_windows
from src.features import extract_window_features

target_dir = r"C:\103NZ_30"
out_dir = r"C:\Medical\dataset\features_103nz"
os.makedirs(out_dir, exist_ok=True)

ffmpeg_bin = r"C:\Medical\.venv\Scripts\ffmpeg.exe"

# Proportional weights for 13 Foley steps:
# 1: Intro (5%), 2: Consent (5%), 3: Instruments (7%), 4: Gloves (9%),
# 5: Privacy (7%), 6: Painting (13%), 7: Lignocaine (10%), 8: Insertion (16%),
# 9: Urobag (8%), 10: Balloon (8%), 11: Traction (4%), 12: Prepuce (4%), 13: Dressing (4%)
weights = np.array([0.05, 0.05, 0.07, 0.09, 0.07, 0.13, 0.10, 0.16, 0.08, 0.08, 0.04, 0.04, 0.04])
weights = weights / np.sum(weights)

for name in ["014.MP4", "018.MP4", "019.MP4", "020.MP4"]:
    stem = os.path.splitext(name)[0]
    vpath = os.path.join(target_dir, name)
    wpath = os.path.join(out_dir, f"{stem}.wav")
    npz_path = os.path.join(out_dir, f"{stem}_feat.npz")

    # 1. Extract audio
    cmd = [
        ffmpeg_bin, "-y", "-i", vpath,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        wpath
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # 2. Load audio and video frames
    audio, sr = load_wav_audio(wpath, target_sr=16000)
    frames, fps = load_video_frames(vpath, target_fps=2, frame_size=(64, 64))
    dur_sec = len(audio) / sr

    # 3. Slice windows
    windows = slice_windows(audio, sr, frames, fps, window_sec=4.0, hop_sec=2.0)
    n_wins = len(windows)

    # 4. Construct step sequence proportional to duration
    # Every window midpoint maps to a step 1..13
    y = np.zeros(n_wins, dtype=np.int64)
    cur_idx = 0
    for sid in range(1, 14):
        dur_w = int(round(weights[sid - 1] * n_wins))
        end_idx = min(cur_idx + dur_w, n_wins) if sid < 13 else n_wins
        y[cur_idx:end_idx] = sid
        cur_idx = end_idx

    # Ensure every step has at least 1 window
    for sid in range(1, 14):
        if sid not in y:
            # allocate at least one window
            idx = int(round((sid - 0.5) / 13.0 * n_wins))
            y[min(idx, n_wins - 1)] = sid

    # 5. Extract multimodal features
    feats = []
    for w in windows:
        f = extract_window_features(w["video"], w["audio"], audio_sr=sr)
        feats.append(f)
    X = np.array(feats, dtype=np.float32)

    np.savez_compressed(npz_path, X=X, y=y)
    print(f"[{stem}] Extracted X={X.shape}, y={y.shape}, duration={dur_sec:.1f}s, steps present={sorted(list(set(y)))}")

    if os.path.exists(wpath):
        os.remove(wpath)

print("\nFeature extraction for C:\\103NZ_30 complete!")
