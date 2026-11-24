"""Temporal Progression and Harmonic Positional Encodings.

D Y Patil Medical College - OSCE Procedural Skills Assessment Model
Pure Python Standard Library + NumPy ONLY.

Provides normalized procedural progression, harmonic Fourier encodings,
and pacing factors to anchor neural models to the clinical temporal prior.
"""

import numpy as np
from typing import Tuple


COHORT_MEDIAN_DURATION_SEC = 158.0


def compute_temporal_positional_features(
    window_idx: int,
    total_windows: int,
    window_sec: float = 4.0,
    hop_sec: float = 2.0,
    expected_duration_sec: float = COHORT_MEDIAN_DURATION_SEC,
) -> np.ndarray:
    """Compute 8-dimensional temporal progression features for a single window.

    Features:
    0: Normalized procedural progress [0.0, 1.0]
    1: sin(2 * pi * t_rel) (first harmonic)
    2: cos(2 * pi * t_rel)
    3: sin(4 * pi * t_rel) (second harmonic)
    4: cos(4 * pi * t_rel)
    5: Normalized elapsed seconds (t_mid / 300.0)
    6: Relative session pacing factor (total_duration / cohort_median)
    7: Normalized window index (i / N)
    """
    total_dur_sec = max(1.0, (total_windows * hop_sec) + (window_sec - hop_sec))
    t_start = window_idx * hop_sec
    t_mid = t_start + (window_sec / 2.0)
    t_rel = np.clip(t_mid / total_dur_sec, 0.0, 1.0)

    pacing_factor = total_dur_sec / expected_duration_sec

    feats = np.array([
        t_rel,
        np.sin(2.0 * np.pi * t_rel),
        np.cos(2.0 * np.pi * t_rel),
        np.sin(4.0 * np.pi * t_rel),
        np.cos(4.0 * np.pi * t_rel),
        min(1.0, t_mid / 300.0),
        pacing_factor,
        window_idx / max(1, total_windows)
    ], dtype=np.float32)

    return feats


def add_temporal_embeddings_to_session(X: np.ndarray, hop_sec: float = 2.0, window_sec: float = 4.0) -> np.ndarray:
    """Augment an (N, D) feature array with 8-D temporal positional embeddings."""
    N, D = X.shape
    t_feats = np.zeros((N, 8), dtype=np.float32)
    for i in range(N):
        t_feats[i] = compute_temporal_positional_features(i, N, window_sec=window_sec, hop_sec=hop_sec)
    return np.hstack([X, t_feats])
