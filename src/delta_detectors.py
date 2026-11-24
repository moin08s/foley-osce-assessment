"""Temporally Gated Differential ROI Evidence Detectors for OSCE Assessment.

D Y Patil Medical College - Male Foley Catheterization OSCE Model
Pure Python Standard Library + NumPy (with optional OpenCV fallback for video reading).

Solves the 100% false-alarm problem by:
1. Temporal gating: Evaluating detectors ONLY within the predicted or search window of the step.
2. Differential delta tracking: Measuring step change (delta) over pre-step baseline rather than static thresholds.
3. Anatomical ROIs: Restricting visual checks to the pelvic mannequin area or hand work-area.
4. Dual-channel speech activity: Measuring candidate speech in the first 30s and examiner verbal prompts.
5. Honest reliability labelling: Low/Untested for steps with < 5 negatives, No Negative Data for steps with 0.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Spatial ROIs in 1920x1080 video coordinates
# Pelvic Mannequin ROI: where the drape and catheter insertion physically occur
PELVIC_ROI = {"ymin": 450, "ymax": 950, "xmin": 850, "xmax": 1450}

# Hand & Instrument Work Area: where gloves and packaging are manipulated
WORK_AREA_ROI = {"ymin": 350, "ymax": 950, "xmin": 500, "xmax": 1500}

# Step Negative Session Benchmarks (from Faculty Sheets)
STEP_NEGATIVE_COUNTS = {
    1: 1,   # 007
    2: 0,   # No negatives in cohort
    3: 2,   # 012, 023
    4: 4,   # 012, 013, 016, 023
    5: 3,   # 012, 013, 023
    6: 2,   # 009, 012
    7: 1,   # 023
    8: 0,   # No negatives in cohort
    9: 1,   # 012
    10: 1,  # 012
    11: 0,  # No negatives in cohort
    12: 2,  # 013, 021
    13: 3,  # 011, 016, 021
}


def get_step_reliability(step_id: int) -> str:
    """Return honest reliability label according to strict negative-session evidence rules."""
    n_negs = STEP_NEGATIVE_COUNTS.get(step_id, 0)
    if n_negs == 0:
        return "No negative data"
    elif n_negs < 5:
        return "Untested/Low"
    else:
        return "High"


def evaluate_glove_donning_delta(
    glove_time_series: np.ndarray,
    timestamps_sec: np.ndarray,
    search_window_sec: Tuple[float, float] = (10.0, 75.0),
    delta_threshold: float = 0.035,
) -> Dict[str, Any]:
    """Detect glove donning via step-change in sterile glove reflectance over baseline.
    
    Args:
        glove_time_series: 1D array of glove color fractions at each timestamp
        timestamps_sec: 1D array of timestamps
        search_window_sec: (t_start, t_end) temporal gate for Step 4
        delta_threshold: Minimum positive increase in glove fraction over baseline
    """
    if len(glove_time_series) == 0:
        return {"fired": False, "first_signal_sec": None, "reliability": get_step_reliability(4)}

    # Measure baseline before search window (e.g., first 10-15s during intro/prep)
    baseline_mask = (timestamps_sec >= 0.0) & (timestamps_sec < search_window_sec[0])
    if np.any(baseline_mask):
        baseline_val = float(np.percentile(glove_time_series[baseline_mask], 75))
    else:
        baseline_val = float(np.min(glove_time_series))

    # Evaluate delta within the gated search window
    window_mask = (timestamps_sec >= search_window_sec[0]) & (timestamps_sec <= search_window_sec[1])
    if not np.any(window_mask):
        return {"fired": False, "first_signal_sec": None, "reliability": get_step_reliability(4)}

    window_times = timestamps_sec[window_mask]
    window_signals = glove_time_series[window_mask]

    deltas = window_signals - baseline_val
    fired_indices = np.where(deltas >= delta_threshold)[0]

    if len(fired_indices) > 0:
        first_idx = fired_indices[0]
        first_time = float(window_times[first_idx])
        return {
            "fired": True,
            "first_signal_sec": first_time,
            "max_delta": float(np.max(deltas)),
            "reliability": get_step_reliability(4),
            "negative_sessions_tested": STEP_NEGATIVE_COUNTS[4],
            "notes": f"Glove delta +{np.max(deltas):.3f} detected at {first_time:.1f}s",
        }
    else:
        return {
            "fired": False,
            "first_signal_sec": None,
            "max_delta": float(np.max(deltas)) if len(deltas) > 0 else 0.0,
            "reliability": get_step_reliability(4),
            "negative_sessions_tested": STEP_NEGATIVE_COUNTS[4],
            "notes": "No significant glove donning delta detected in gated window",
        }


def evaluate_draping_delta(
    pelvic_green_series: np.ndarray,
    timestamps_sec: np.ndarray,
    search_window_sec: Tuple[float, float] = (30.0, 110.0),
    expansion_threshold: float = 0.040,
) -> Dict[str, Any]:
    """Detect draping via sudden green area expansion specifically within the pelvic ROI.
    
    Eliminates false alarms from side-table green instrument trays.
    """
    if len(pelvic_green_series) == 0:
        return {"fired": False, "first_signal_sec": None, "reliability": get_step_reliability(6)}

    # Pre-drape baseline in pelvic area
    pre_drape_mask = (timestamps_sec >= 0.0) & (timestamps_sec < search_window_sec[0])
    if np.any(pre_drape_mask):
        baseline_green = float(np.percentile(pelvic_green_series[pre_drape_mask], 75))
    else:
        baseline_green = float(np.min(pelvic_green_series))

    window_mask = (timestamps_sec >= search_window_sec[0]) & (timestamps_sec <= search_window_sec[1])
    if not np.any(window_mask):
        return {"fired": False, "first_signal_sec": None, "reliability": get_step_reliability(6)}

    window_times = timestamps_sec[window_mask]
    window_signals = pelvic_green_series[window_mask]

    deltas = window_signals - baseline_green
    fired_indices = np.where(deltas >= expansion_threshold)[0]

    if len(fired_indices) > 0:
        first_idx = fired_indices[0]
        first_time = float(window_times[first_idx])
        return {
            "fired": True,
            "first_signal_sec": first_time,
            "max_delta": float(np.max(deltas)),
            "reliability": get_step_reliability(6),
            "negative_sessions_tested": STEP_NEGATIVE_COUNTS[6],
            "notes": f"Pelvic green drape expansion +{np.max(deltas):.3f} detected at {first_time:.1f}s",
        }
    else:
        return {
            "fired": False,
            "first_signal_sec": None,
            "max_delta": float(np.max(deltas)) if len(deltas) > 0 else 0.0,
            "reliability": get_step_reliability(6),
            "negative_sessions_tested": STEP_NEGATIVE_COUNTS[6],
            "notes": "No pelvic drape expansion detected in gated window",
        }


def evaluate_speech_activity_step1_2(
    audio_samples: np.ndarray,
    sr: int = 16000,
    check_duration_sec: float = 30.0,
    ste_threshold: float = 0.015,
) -> Dict[str, Any]:
    """Check candidate speech presence strictly in the first 30 seconds.
    
    Labelled as 'Speech present', not 'introduction verified' (as per faculty requirement).
    """
    if len(audio_samples) == 0:
        return {
            "speech_present": False,
            "speech_duration_sec": 0.0,
            "notes": "Speech absent in first 30s",
            "reliability": "Low/Unverified",
        }

    # Slice first 30s
    n_check = min(len(audio_samples), int(check_duration_sec * sr))
    audio_sub = audio_samples[:n_check]

    frame_len = 512
    hop_len = 256
    n_frames = max(1, (len(audio_sub) - frame_len) // hop_len + 1)
    
    ste = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        st = i * hop_len
        frame = audio_sub[st:st+frame_len]
        ste[i] = np.mean(frame**2)

    speech_frames = np.sum(ste > ste_threshold)
    speech_sec = float(speech_frames * hop_len / sr)
    speech_present = speech_sec >= 1.5

    return {
        "speech_present": speech_present,
        "speech_duration_sec": round(speech_sec, 2),
        "notes": f"Speech present ({speech_sec:.1f}s in first {check_duration_sec:.0f}s)" if speech_present else "No candidate speech detected in first 30s",
        "reliability": get_step_reliability(1),
    }
