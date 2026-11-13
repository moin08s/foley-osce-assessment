"""Data Loader for OSCE Male Foley Catheterization Assessment.

D Y Patil Medical College - Procedural Skills Assessment Model
Pure Python Standard Library + NumPy implementation (no external ML frameworks).

Functions:
- read_manifest: Parse manifest/labels CSV into structured records.
- load_wav_audio: Read 16 kHz mono 16-bit PCM WAV using Python's `wave` module.
- load_video_frames: Stream raw RGB24 video frames using FFmpeg rawvideo pipe.
- slice_windows: Cut time-aligned 2-6s audio & video windows with 50% overlap.
- label_window: Determine ground truth step label (0 = background, 1-13 = OSCE steps).
"""

import csv
import os
import shutil
import subprocess
import wave
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# Clinical step metadata mapping Step ID to standard procedural names
STEP_LABELS = {
    0: "background",
    1: "introduce_explain",
    2: "informed_consent",
    3: "check_instruments",
    4: "hand_hygiene_gloves",
    5: "privacy_position_exposure",
    6: "painting_draping",
    7: "lignocaine_jelly",
    8: "open_catheter_insert",
    9: "attach_urobag_confirm_urine",
    10: "inflate_balloon",
    11: "traction_until_resistance",
    12: "reposition_prepuce",
    13: "tip_dressing",
}

LABEL_TO_STEP = {v: k for k, v in STEP_LABELS.items()}


def get_ffmpeg_path() -> str:
    """Find the path to the ffmpeg executable on this system."""
    # 1. Check local project virtualenv
    local_venv_ffmpeg = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts", "ffmpeg.exe"))
    if os.path.isfile(local_venv_ffmpeg):
        return local_venv_ffmpeg

    # 2. Check system PATH
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    # 3. Fallback standard default
    return "ffmpeg"


def read_manifest(manifest_path: str) -> List[Dict[str, Union[str, float, int]]]:
    """Read annotation manifest CSV.
    
    Expected CSV columns:
    clip_id, session_id, student_id, split, video_path, audio_path,
    step_id, step_label, start_sec, end_sec, correct(Y/N), error_type,
    prompt_given(Y/N), speech_transcript, speech_present(Y/N), audio_notes, ...
    """
    records = []
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = dict(row)
            try:
                cleaned["step_id"] = int(cleaned.get("step_id", 0))
            except (ValueError, TypeError):
                cleaned["step_id"] = 0
            try:
                cleaned["start_sec"] = float(cleaned.get("start_sec", 0.0))
            except (ValueError, TypeError):
                cleaned["start_sec"] = 0.0
            try:
                cleaned["end_sec"] = float(cleaned.get("end_sec", 0.0))
            except (ValueError, TypeError):
                cleaned["end_sec"] = 0.0

            records.append(cleaned)
    return records


def load_wav_audio(wav_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load an uncompressed WAV file using Python's standard library `wave` module.
    
    Converts 16-bit PCM integer samples to float32 in [-1.0, 1.0].
    Validates sample rate and handles mono/stereo downmixing.
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV audio file not found: {wav_path}")

    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:  # 16-bit PCM
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:  # 8-bit unsigned
        samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:  # 32-bit int
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    # Downmix to mono if multi-channel
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample to target_sr using linear interpolation if necessary
    if framerate != target_sr:
        duration = len(samples) / framerate
        target_len = int(round(duration * target_sr))
        orig_indices = np.linspace(0, len(samples) - 1, target_len)
        samples = np.interp(orig_indices, np.arange(len(samples)), samples)
        framerate = target_sr

    return samples.astype(np.float32), framerate


def load_video_frames(
    video_path: str,
    target_fps: int = 4,
    frame_size: Tuple[int, int] = (64, 64),
    max_duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Extract downsampled RGB video frames using an FFmpeg rawvideo pipe.
    
    Why FFmpeg:
    In compliance with the project constraint (no OpenCV/PyTorch/TensorFlow),
    FFmpeg is used purely as a standalone system binary to decode MP4/H.264
    into raw uncompressed RGB24 bytes piped directly to stdout, which Python
    reads into NumPy. No third-party video libraries are required.
    
    Args:
        video_path: Path to .mp4 or .avi file
        target_fps: Frames per second to decode (e.g. 4 fps is fast and sufficient for motion)
        frame_size: (width, height) tuple, defaults to 64x64
        max_duration_sec: Optional duration cap in seconds
        
    Returns:
        frames: np.ndarray of shape (N_frames, height, width, 3), uint8
        fps: effective frames per second
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffmpeg_bin = get_ffmpeg_path()
    w, h = frame_size

    cmd = [
        ffmpeg_bin,
        "-v", "error",
        "-i", video_path,
    ]
    if max_duration_sec is not None:
        cmd.extend(["-t", str(max_duration_sec)])
    cmd.extend([
        "-vf", f"fps={target_fps},scale={w}:{h}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-"
    ])

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**7,
    )
    raw_bytes, stderr_bytes = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with error: {stderr_bytes.decode('utf-8', errors='ignore')}")

    bytes_per_frame = w * h * 3
    num_frames = len(raw_bytes) // bytes_per_frame
    if num_frames == 0:
        raise ValueError(f"No frames decoded from video {video_path}")

    usable_bytes = raw_bytes[: num_frames * bytes_per_frame]
    frames = np.frombuffer(usable_bytes, dtype=np.uint8).reshape((num_frames, h, w, 3))
    return frames, float(target_fps)


def label_window(
    w_start: float,
    w_end: float,
    manifest_records: List[Dict[str, Union[str, float, int]]],
) -> int:
    """Assign an OSCE step label (0-13) to a window based on ground truth overlap.
    
    A window is labeled with step N if its midpoint falls within the step's
    annotated time interval [start_sec, end_sec]. If no step matches, it is
    classified as background (0).
    """
    w_mid = (w_start + w_end) / 2.0
    for rec in manifest_records:
        s_start = float(rec["start_sec"])
        s_end = float(rec["end_sec"])
        if s_start <= w_mid < s_end:
            return int(rec["step_id"])
    return 0


def slice_windows(
    audio: np.ndarray,
    audio_sr: int,
    video: np.ndarray,
    video_fps: float,
    window_sec: float = 4.0,
    hop_sec: float = 2.0,
    manifest_records: Optional[List[Dict[str, Union[str, float, int]]]] = None,
) -> List[Dict[str, Union[np.ndarray, int, float]]]:
    """Slice synchronized audio and video into sliding windows with 50% overlap.
    
    Args:
        audio: 1D float array of 16 kHz audio samples
        audio_sr: Audio sampling rate (Hz)
        video: 4D uint8 array (N_frames, H, W, 3)
        video_fps: Video frame rate (fps)
        window_sec: Window length in seconds (default 4.0 s)
        hop_sec: Step hop in seconds (default 2.0 s, 50% overlap)
        manifest_records: Optional list of step annotations for ground truth labeling
        
    Returns:
        List of window dictionaries containing:
        - "audio": 1D array of audio samples in this window
        - "video": 4D array of video frames in this window
        - "start_sec": Window start time (s)
        - "end_sec": Window end time (s)
        - "label": Step label integer 0-13 (0 if manifest_records is None)
    """
    total_audio_sec = len(audio) / audio_sr if audio_sr > 0 else 0.0
    total_video_sec = len(video) / video_fps if video_fps > 0 else 0.0
    total_duration = min(total_audio_sec, total_video_sec)

    windows = []
    w_start = 0.0
    while w_start + window_sec <= total_duration + (hop_sec * 0.5):
        w_end = min(w_start + window_sec, total_duration)
        if w_end - w_start < 1.0:
            break

        # Audio slice
        a_idx_start = int(round(w_start * audio_sr))
        a_idx_end = int(round(w_end * audio_sr))
        audio_slice = audio[a_idx_start:a_idx_end]
        target_audio_len = int(round(window_sec * audio_sr))
        if len(audio_slice) < target_audio_len:
            pad = np.zeros(target_audio_len - len(audio_slice), dtype=audio.dtype)
            audio_slice = np.concatenate([audio_slice, pad])
        elif len(audio_slice) > target_audio_len:
            audio_slice = audio_slice[:target_audio_len]

        # Video slice
        v_idx_start = int(round(w_start * video_fps))
        v_idx_end = int(round(w_end * video_fps))
        video_slice = video[v_idx_start:v_idx_end]
        target_video_len = int(round(window_sec * video_fps))
        if len(video_slice) < target_video_len:
            if len(video_slice) > 0:
                repeat_last = np.repeat(video_slice[-1:], target_video_len - len(video_slice), axis=0)
                video_slice = np.concatenate([video_slice, repeat_last], axis=0)
            else:
                video_slice = np.zeros((target_video_len, 64, 64, 3), dtype=np.uint8)
        elif len(video_slice) > target_video_len:
            video_slice = video_slice[:target_video_len]

        label = 0
        if manifest_records:
            label = label_window(w_start, w_end, manifest_records)

        windows.append({
            "audio": audio_slice,
            "video": video_slice,
            "start_sec": round(w_start, 2),
            "end_sec": round(w_end, 2),
            "label": label,
        })

        w_start += hop_sec

    return windows
