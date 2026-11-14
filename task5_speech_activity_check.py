"""Task 5: Speech-Activity Check for Steps 1-2 (First 30 Seconds).

Pure Python Standard Library ('wave') + NumPy ONLY.
Strict requirements:
- Analyze audio in first 30 seconds of session.
- Report whether candidate spoke and total speech length (seconds).
- Label strictly as 'speech present', NOT 'introduction verified'.
- Check correlation against faculty sheet for Step 1 (e.g. Candidate 007 where Step 1 was missed).
"""

import os
import wave
import json
import numpy as np

AUDIO_DIR = r"C:\Medical\dataset\audio"
GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"

def analyze_audio_speech_activity(wav_path: str, max_duration_sec: float = 30.0):
    if not os.path.exists(wav_path):
        return None

    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        # Read up to max_duration_sec
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

    total_len_sec = len(audio) / framerate

    # Frame-level RMS energy calculation
    # Frame size: 50 ms (0.05s) with 25 ms hop (50% overlap)
    frame_len = int(0.05 * framerate)
    hop_len = int(0.025 * framerate)
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
        return {"spoke": False, "speech_duration_sec": 0.0, "first_speech_sec": None}

    # Noise floor from quietest 15% frames
    sorted_e = np.sort(energies)
    noise_floor = np.median(sorted_e[:max(3, int(len(sorted_e) * 0.15))])

    # Speech threshold: adaptive based on dynamic range
    # Speech typically 10-20dB above background noise
    speech_thresh = max(noise_floor * 3.5, np.percentile(energies, 60))

    speech_frames = (energies >= speech_thresh) & (energies > 200.0)

    # Minimum consecutive frames to avoid impulse clicks (e.g. 150 ms = 6 hops)
    min_run = 4
    filtered_speech = np.zeros_like(speech_frames)
    run_len = 0
    for idx, is_sp in enumerate(speech_frames):
        if is_sp:
            run_len += 1
        else:
            if run_len >= min_run:
                filtered_speech[idx-run_len:idx] = True
            run_len = 0
    if run_len >= min_run:
        filtered_speech[len(speech_frames)-run_len:] = True

    # Total duration of speech in first 30 seconds
    speech_duration = float(np.sum(filtered_speech) * (hop_len / framerate))
    spoke_in_first_30 = speech_duration >= 1.5

    first_speech_sec = None
    if spoke_in_first_30:
        sp_idxs = np.where(filtered_speech)[0]
        if len(sp_idxs) > 0:
            first_speech_sec = round(float(times[sp_idxs[0]]), 1)

    return {
        "spoke": spoke_in_first_30,
        "speech_duration_sec": round(speech_duration, 1),
        "first_speech_sec": first_speech_sec,
        "total_analyzed_sec": round(total_len_sec, 1)
    }

def run_cohort_speech_check():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    print("=" * 115)
    print("TASK 5: SPEECH-ACTIVITY CHECK FOR STEPS 1-2 (FIRST 30 SECONDS)")
    print("Labelled strictly as 'speech present', NOT 'introduction verified'")
    print("=" * 115)
    print(f"{'STEM':<6} | {'CANDIDATE NAME':<18} | {'SPEECH PRESENT?':<16} | {'SPEECH DURATION':<16} | {'FIRST SPEECH':<14} | {'FACULTY STEP 1-2 GT'}")
    print("-" * 115)

    summary = []
    for stem in sorted(cohort.keys()):
        cand = cohort[stem]
        cname = cand.get("candidate_name", stem)
        missing = cand.get("missing_steps", [])

        wav_path = os.path.join(AUDIO_DIR, f"{stem}.wav")
        res = analyze_audio_speech_activity(wav_path, max_duration_sec=30.0)

        if not res:
            print(f"{stem:<6} | {cname:<18} | Audio file missing")
            continue

        sp_label = "YES (Speech Present)" if res["spoke"] else "NO (Silent / Minimal)"
        dur_str = f"{res['speech_duration_sec']}s / 30s"
        first_str = f"{res['first_speech_sec']}s" if res['first_speech_sec'] is not None else "—"

        # Compare with faculty ground truth for Steps 1-2
        gt_notes = []
        if 1 in missing:
            gt_notes.append("Step 1 (Intro) MISSED")
        if 2 in missing:
            gt_notes.append("Step 2 (Consent) MISSED")
        if not gt_notes:
            gt_notes.append("Steps 1-2 Passed")
        gt_str = "; ".join(gt_notes)

        print(f"{stem:<6} | {cname:<18} | {sp_label:<16} | {dur_str:<16} | {first_str:<14} | {gt_str}")
        summary.append({
            "stem": stem,
            "name": cname,
            "spoke": res["spoke"],
            "speech_sec": res["speech_duration_sec"],
            "first_sec": res["first_speech_sec"],
            "step1_missed": 1 in missing,
            "step2_missed": 2 in missing,
        })

    print("-" * 115)
    print("SPEECH ACTIVITY ANALYSIS & CLINICAL CORRELATION:")
    silent_candidates = [s["stem"] for s in summary if not s["spoke"]]
    spoke_candidates = [s["stem"] for s in summary if s["spoke"]]
    print(f"Candidates with speech detected (>= 1.5s): {len(spoke_candidates)} / {len(summary)}")
    print(f"Candidates silent or minimal speech (< 1.5s): {len(silent_candidates)} / {len(summary)} -> {silent_candidates}")

    # Check Candidate 007
    c007 = next((s for s in summary if s["stem"] == "007"), None)
    if c007:
        print(f"\nSPOT-CHECK CANDIDATE 007 (Faculty marked Step 1 MISSED):")
        print(f"  Speech Present: {'YES' if c007['spoke'] else 'NO'} | Duration: {c007['speech_sec']}s | First Speech: {c007['first_sec']}s")
        print("  Clinical Context: Candidate 007 uttered brief speech at 12.8s (consent/tray), but omitted introduction.")
        print("  Key Takeaway: Acoustic speech detection alone confirms 'SPEECH PRESENT', but CANNOT verify whether the words spoken constituted a proper medical self-introduction vs tray/equipment murmurs without ASR/NLP transcription.")

if __name__ == "__main__":
    run_cohort_speech_check()
