"""Speech Activity Detector for Steps 1 and 2 (First 30 Seconds).

D Y Patil Medical College - OSCE Procedural Skills Assistant
Pure Python Standard Library ('wave') + NumPy ONLY.

Strict clinical requirement:
- Labels detected vocalization strictly as 'SPEECH PRESENT', NOT 'Introduction Verified'.
- Computes speech duration and timestamp of first vocalization.
"""

import os
import wave
from typing import Dict, Any, Optional
import numpy as np


def detect_speech_activity(wav_path: str, max_duration_sec: float = 30.0) -> Dict[str, Any]:
    """Detect voice activity in the first N seconds of a session audio track."""
    if not os.path.isfile(wav_path):
        return {
            "speech_present": False,
            "speech_duration_sec": 0.0,
            "first_speech_sec": None,
            "label": "No Audio File"
        }

    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        frames_to_read = min(n_frames, int(framerate * max_duration_sec))
        raw_bytes = wf.readframes(frames_to_read)

    if sampwidth == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    elif sampwidth == 1:
        audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128) * 256.0
    else:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 65536.0

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    frame_len = int(0.05 * framerate) # 50ms
    hop_len = int(0.025 * framerate)  # 25ms
    n_hops = max(1, (len(audio) - frame_len) // hop_len)

    energies = []
    times = []
    for i in range(n_hops):
        start = i * hop_len
        end = start + frame_len
        frame = audio[start:end]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        energies.append(rms)
        times.append(start / framerate)

    energies = np.array(energies)
    times = np.array(times)

    if len(energies) == 0:
        return {"speech_present": False, "speech_duration_sec": 0.0, "first_speech_sec": None, "label": "Silence"}

    sorted_e = np.sort(energies)
    noise_floor = np.median(sorted_e[:max(3, int(len(sorted_e) * 0.15))])
    speech_thresh = max(noise_floor * 3.5, np.percentile(energies, 60))

    speech_frames = (energies >= speech_thresh) & (energies > 200.0)

    # Filter out transient clicks (require >= 4 consecutive frames = 100ms)
    min_run = 4
    filtered_speech = np.zeros_like(speech_frames)
    run_len = 0
    for idx, is_sp in enumerate(speech_frames):
        if is_sp:
            run_len += 1
        else:
            if run_len >= min_run:
                filtered_speech[idx - run_len:idx] = True
            run_len = 0
    if run_len >= min_run:
        filtered_speech[len(speech_frames) - run_len:] = True

    speech_duration = float(np.sum(filtered_speech) * (hop_len / framerate))
    spoke_in_first_30 = speech_duration >= 1.5

    first_speech_sec = None
    if spoke_in_first_30:
        sp_idxs = np.where(filtered_speech)[0]
        if len(sp_idxs) > 0:
            first_speech_sec = round(float(times[sp_idxs[0]]), 1)

    return {
        "speech_present": spoke_in_first_30,
        "speech_duration_sec": round(speech_duration, 2),
        "first_speech_sec": first_speech_sec,
        "label": "Speech Present" if spoke_in_first_30 else "No Speech Detected",
        "clinical_note": "Acoustic detection verifies speech presence only; content must be faculty-verified."
    }
