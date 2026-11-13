"""Procedural Video Slicing Engine with Synchronized Audio & Defect Detection.

D Y Patil Medical College - Procedural Skills Assessment Model
Pure Python Standard Library (subprocess) + FFmpeg.
Zero third-party ML/DL or video editing dependencies.

Functions:
- slice_step_clip: Extracts a browser-compatible H.264 + AAC sub-clip with synchronized audio.
- generate_poster_frame: Extracts a companion JPEG thumbnail for instant UI visibility.
- audit_step_defects: Automatically evaluates the clip for clinical/procedural flaws.
- slice_session_steps: Carves all 13 procedural steps into dedicated evidence clips with audio.
"""

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from .data_loader import STEP_LABELS, get_ffmpeg_path


def slice_step_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    include_audio: bool = True,
) -> bool:
    """Extract an evidence sub-clip between start_sec and end_sec WITH AUDIO.
    
    Uses ultrafast H.264 encode with faststart header so the clip starts
    playing immediately in browsers without buffering the entire file.
    
    Args:
        video_path: Path to source candidate video (.mp4/.avi/.webm)
        start_sec: Step start timestamp (s)
        end_sec: Step end timestamp (s)
        output_path: Destination path for the sliced .mp4
        include_audio: Whether to preserve synchronized audio (ALWAYS True for OSCE)
        
    Returns:
        bool: True if sliced file was successfully generated.
    """
    if not os.path.isfile(video_path):
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    duration = max(0.5, end_sec - start_sec)
    ffmpeg_bin = get_ffmpeg_path()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]

    if include_audio:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "16000",
        ])
    else:
        cmd.append("-an")

    cmd.append(output_path)

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return os.path.isfile(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        print(f"[VideoSlicer] Slicing error for {output_path}: {e}")
        return False


def generate_poster_frame(video_path: str, timestamp_sec: float, output_jpg: str) -> bool:
    """Extract a single high-quality JPEG poster frame for video thumbnail visibility."""
    if not os.path.isfile(video_path):
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_jpg)), exist_ok=True)
    ffmpeg_bin = get_ffmpeg_path()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss", f"{timestamp_sec:.3f}",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_jpg
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return os.path.isfile(output_jpg) and os.path.getsize(output_jpg) > 500
    except Exception:
        return False


def audit_step_defects(
    step_id: int,
    step_label: str,
    duration_sec: float,
    all_detected_order: List[int],
    speech_vad_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Analyze step performance for procedural or clinical defects.
    
    Returns structured defect audit:
    - has_defect: bool
    - defect_level: 'NONE' | 'INFO' | 'COMMUNICATION' | 'TECHNIQUE' | 'CRITICAL'
    - defect_summary: Short headline
    - clinical_notes: Guidance for the reviewing teacher
    """
    has_defect = False
    defect_level = "NONE"
    summary = "Technique Verified Correct"
    notes = "Step executed according to standard surgical checklist."

    # Step 1: Introduction Communication
    if step_id == 1:
        if speech_vad_ratio is not None and speech_vad_ratio < 0.05:
            has_defect = True
            defect_level = "COMMUNICATION"
            summary = "Defect: Missing Verbal Introduction"
            notes = "Student did not verbally introduce self or explain procedure to patient (silent audio)."
        elif duration_sec < 4.0:
            has_defect = True
            defect_level = "TECHNIQUE"
            summary = "Defect: Rushed Introduction"
            notes = f"Introduction was very brief ({duration_sec:.1f}s); may lack adequate patient orientation."

    # Step 2: Informed Consent Communication
    elif step_id == 2:
        if speech_vad_ratio is not None and speech_vad_ratio < 0.05:
            has_defect = True
            defect_level = "COMMUNICATION"
            summary = "Defect: Missing Verbal Consent"
            notes = "Student did not ask for or confirm patient consent verbally."
        elif 4 in all_detected_order and all_detected_order.index(2) > all_detected_order.index(4):
            has_defect = True
            defect_level = "CRITICAL"
            summary = "Defect: Consent Asked After Donning Sterile Gloves"
            notes = "Informed consent was sought after sterile gloves were worn (procedural order violation)."

    # Step 4: Hand Hygiene & Sterile Gloves
    elif step_id == 4:
        if 6 in all_detected_order and all_detected_order.index(4) > all_detected_order.index(6):
            has_defect = True
            defect_level = "CRITICAL"
            summary = "CRITICAL DEFECT: Aseptic Breach (Painting Before Gloves)"
            notes = "Antiseptic painting commenced prior to donning sterile gloves."
        elif duration_sec < 6.0:
            has_defect = True
            defect_level = "TECHNIQUE"
            summary = "Defect: Inadequate Hand Hygiene Dwell Time"
            notes = f"Sanitization & glove donning lasted only {duration_sec:.1f}s."

    # Step 7: Lignocaine Jelly Anesthetic Dwell Time
    elif step_id == 7:
        if duration_sec < 8.0:
            has_defect = True
            defect_level = "TECHNIQUE"
            summary = "Defect: Insufficient Jelly Dwell Time"
            notes = f"Lignocaine jelly instillation & wait lasted only {duration_sec:.1f}s (minimum 10-15s recommended for urethral anesthesia)."

    # Step 9 & 10: Balloon Inflation & Urine Confirmation (CRITICAL SAFETY)
    elif step_id == 10:
        if 9 not in all_detected_order or all_detected_order.index(10) < all_detected_order.index(9):
            has_defect = True
            defect_level = "CRITICAL"
            summary = "CRITICAL SAFETY DEFECT: Balloon Inflated Before Urine Confirmed"
            notes = "High risk of inflating catheter retention balloon inside prostatic urethra causing severe urethral rupture!"

    # Step 12: Reposition Prepuce (PARAPHIMOSIS SAFETY)
    elif step_id == 12:
        if duration_sec < 2.0:
            has_defect = True
            defect_level = "CRITICAL"
            summary = "CRITICAL DEFECT: Foreskin Repositioning Questionable/Omitted"
            notes = "Failure to draw prepuce forward over glans penis causes venous congestion and acute paraphimosis."

    # Step 13: Tip Dressing terminal order check
    elif step_id == 13:
        if len(all_detected_order) > 0 and all_detected_order[-1] != 13:
            has_defect = True
            defect_level = "INFO"
            summary = "Defect: Tip Dressing Not Final Step"
            notes = "Catheter dressing and securement was followed by subsequent manipulations."

    return {
        "has_defect": has_defect,
        "defect_level": defect_level,
        "defect_summary": summary,
        "clinical_notes": notes,
    }


def slice_session_steps(
    video_path: str,
    detected_segments: List[Dict[str, Any]],
    output_base_dir: str,
    session_id: str,
    speech_activity: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    """Carve individual step video clips with synchronized audio for teacher inspection.
    
    Outputs browser-ready MP4 sub-clips and JPEG thumbnails in:
        {output_base_dir}/slices/{session_id}/step_{sid:02d}_{label}.mp4
        
    Returns list of step clip evidence dictionaries.
    """
    slice_dir = os.path.join(output_base_dir, "slices", session_id)
    os.makedirs(slice_dir, exist_ok=True)

    detected_order = [s["step_id"] for s in detected_segments]
    step_clips = []

    for seg in detected_segments:
        sid = seg["step_id"]
        label = seg.get("step_label", STEP_LABELS.get(sid, f"step_{sid}"))
        s_start = max(0.0, float(seg["start_sec"]))
        s_end = float(seg["end_sec"])
        dur = round(s_end - s_start, 2)
        conf = float(seg.get("confidence", 1.0))

        clip_filename = f"step_{sid:02d}_{label}.mp4"
        thumb_filename = f"step_{sid:02d}_{label}.jpg"
        clip_path = os.path.join(slice_dir, clip_filename)
        thumb_path = os.path.join(slice_dir, thumb_filename)

        # Slice video clip with audio
        slice_success = slice_step_clip(
            video_path=video_path,
            start_sec=s_start,
            end_sec=s_end,
            output_path=clip_path,
            include_audio=True,
        )

        # Generate companion JPEG poster thumbnail
        thumb_time = s_start + min(1.0, dur * 0.5)
        thumb_success = generate_poster_frame(
            video_path=video_path,
            timestamp_sec=thumb_time,
            output_jpg=thumb_path,
        )

        # Audit step for any defect
        vad = speech_activity.get(sid) if speech_activity else None
        defect_audit = audit_step_defects(
            step_id=sid,
            step_label=label,
            duration_sec=dur,
            all_detected_order=detected_order,
            speech_vad_ratio=vad,
        )

        clip_info = {
            "step_id": sid,
            "step_label": label,
            "start_sec": s_start,
            "end_sec": s_end,
            "duration_sec": dur,
            "confidence": conf,
            "clip_filename": clip_filename,
            "clip_url": f"/video/slices/{session_id}/{clip_filename}",
            "thumb_url": f"/video/slices/{session_id}/{thumb_filename}",
            "has_audio": True,
            "slice_success": slice_success,
            "defect_audit": defect_audit,
        }
        step_clips.append(clip_info)

    return step_clips
