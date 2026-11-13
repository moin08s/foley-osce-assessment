"""Inference and OSCE Evaluation Script for Male Foley's Catheterization.

D Y Patil Medical College - Procedural Skills Assessment Model
Pure Python Standard Library + NumPy (no third-party ML frameworks).

Usage:
    python predict.py --video path/to/video.mp4 [--audio path/to/audio.wav] [--student-id KaiV]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

import numpy as np

from src.data_loader import (
    STEP_LABELS,
    get_ffmpeg_path,
    load_video_frames,
    load_wav_audio,
    slice_windows,
)
from src.features import compute_short_time_energy, extract_window_features
from src.grading import OSCEGrader
from src.model import FoleyStepClassifier
from src.session_checker import SessionChecker
from src.video_slicer import slice_session_steps


def extract_audio_from_video(video_path: str, output_wav_path: str) -> None:
    """Extract 16 kHz mono 16-bit PCM WAV from video using FFmpeg."""
    ffmpeg_bin = get_ffmpeg_path()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr.decode('utf-8', errors='ignore')}")


def evaluate_video(
    video_path: str,
    audio_path: Optional[str] = None,
    model_path: str = "models/foley_osce_mlp.npz",
    student_id: str = "Candidate-001",
    session_id: str = "session_001",
    faculty_ground_truth: Optional[Dict[str, Any]] = None,
    window_sec: float = 4.0,
    hop_sec: float = 2.0,
    video_fps: int = 4,
) -> Dict[str, Any]:
    """Run end-to-end procedural assessment on a student's video clip."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Auto-extract audio if not provided
    temp_wav = None
    if audio_path is None or not os.path.isfile(audio_path):
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        extract_audio_from_video(video_path, temp_wav.name)
        audio_path = temp_wav.name

    try:
        # 1. Load audio and video
        audio, sr = load_wav_audio(audio_path, target_sr=16000)
        frames, fps = load_video_frames(video_path, target_fps=video_fps)

        # 2. Slice into overlapping windows
        windows = slice_windows(
            audio=audio,
            audio_sr=sr,
            video=frames,
            video_fps=fps,
            window_sec=window_sec,
            hop_sec=hop_sec,
        )

        if len(windows) == 0:
            raise ValueError("Video duration is too short for sliding window assessment.")

        # 3. Extract fused features
        feats = []
        for w in windows:
            f = extract_window_features(w["video"], w["audio"], audio_sr=sr)
            feats.append(f)
        X = np.array(feats, dtype=np.float32)

        # 4. Predict probabilities with trained model
        model = FoleyStepClassifier.load(model_path)
        probs = model.predict_proba(X)

        # 5. Temporal smoothing and segment extraction
        checker = SessionChecker()
        smoothed_preds = checker.smooth_predictions(probs)
        detected_segments = checker.extract_segments(smoothed_preds, windows, window_probs=probs)

        # 6. Audio speech activity per step (for steps 1 & 2 verification)
        speech_activity = {}
        for seg in detected_segments:
            sid = seg["step_id"]
            if sid in (1, 2):
                start_s, end_s = seg["start_sec"], seg["end_sec"]
                s_idx = int(start_s * sr)
                e_idx = int(end_s * sr)
                step_audio = audio[s_idx:e_idx]
                ste = compute_short_time_energy(step_audio, speech_threshold=0.005)
                # ste[2] is vad_ratio (voice activity ratio)
                speech_activity[sid] = float(ste[2])

        # 7. Check sequence rules and safety
        seq_audit = checker.check_sequence_rules(detected_segments, speech_activity_per_step=speech_activity)

        # 8. Compute OSCE grade and format reports
        grader = OSCEGrader()
        report = grader.compute_grade(
            sequence_check=seq_audit,
            student_id=student_id,
            session_id=session_id,
            detected_segments=detected_segments,
            faculty_ground_truth=faculty_ground_truth,
        )
        # 9. Slice individual step evidence clips with audio and defect audits
        upload_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        step_clips = slice_session_steps(
            video_path=video_path,
            detected_segments=detected_segments,
            output_base_dir=upload_base,
            session_id=session_id,
            speech_activity=speech_activity,
        )
        report["step_clips"] = step_clips

        return report

    finally:
        if temp_wav and os.path.exists(temp_wav.name):
            try:
                os.remove(temp_wav.name)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate Male Foley's Catheterization Video (OSCE Grade)")
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 video")
    parser.add_argument("--audio", type=str, default=None, help="Path to input 16kHz WAV audio (optional)")
    parser.add_argument("--model", type=str, default="models/foley_osce_mlp.npz", help="Path to trained model .npz")
    parser.add_argument("--student-id", type=str, default="Candidate-001", help="Student ID / Name")
    parser.add_argument("--session-id", type=str, default="session_001", help="Session ID")
    parser.add_argument("--out-json", type=str, default=None, help="Path to save JSON evaluation report")
    parser.add_argument("--out-report", type=str, default=None, help="Path to save text scorecard report")
    parser.add_argument("--faculty-grade", type=str, default="P", help="Known faculty grade for sanity benchmarking (P, P-1, R)")
    args = parser.parse_args()

    faculty_gt = {"final_grade": args.faculty_grade} if args.faculty_grade else None

    report = evaluate_video(
        video_path=args.video,
        audio_path=args.audio,
        model_path=args.model,
        student_id=args.student_id,
        session_id=args.session_id,
        faculty_ground_truth=faculty_gt,
    )

    grader = OSCEGrader()
    printable_report = grader.format_text_report(report)
    print("\n" + printable_report + "\n")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Structured JSON assessment saved to: {args.out_json}")

    if args.out_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_report)), exist_ok=True)
        with open(args.out_report, "w", encoding="utf-8") as f:
            f.write(printable_report)
        print(f"Printable scorecard report saved to: {args.out_report}")


if __name__ == "__main__":
    main()
