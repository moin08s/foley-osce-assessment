"""Feature Extraction for Foley Catheterization OSCE Assessment.

D Y Patil Medical College - Procedural Skills Assessment Model
Pure Python Standard Library + NumPy (no librosa, scipy, cv2, torch, etc.).

Features extracted per 2-6s sliding window:
1. Video Features:
   - Frame-difference motion energy (mean pixel delta across time)
   - 8x8 grid block motion (spatial motion heatmap across 64 blocks)
   - Procedural color histograms (drape green, urine yellow, catheter orange, gloves)
   - Downsampled 8x8 spatial grayscale projection (postural/anatomical context)
2. Audio Features:
   - FFT-based Log-Mel filterbank (STFT via np.fft.rfft + triangular mel filters)
   - Short-time energy (STE) for speech vs. silence voice activity detection
   - Spectral flux for transient impulse sounds (glove snaps, syringe clicks, tears)
"""

import math
from typing import Dict, Tuple

import numpy as np


def compute_frame_difference_motion(frames_gray: np.ndarray) -> float:
    """Compute overall motion energy from consecutive frame differences.
    
    Args:
        frames_gray: Grayscale frames array of shape (T, H, W), float32 [0.0, 1.0]
    Returns:
        Scalar mean absolute frame difference
    """
    if len(frames_gray) < 2:
        return 0.0
    diff = np.abs(np.diff(frames_gray, axis=0))
    return float(np.mean(diff))


def compute_grid_block_motion(frames_gray: np.ndarray, grid_size: int = 8) -> np.ndarray:
    """Compute 8x8 grid block motion vector magnitudes.
    
    Divides the frame into an 8x8 grid of blocks. For each block,
    computes the mean absolute difference between consecutive frames,
    averaged over time.
    
    Args:
        frames_gray: (T, H, W) float32 [0.0, 1.0]
        grid_size: Number of blocks along each axis (default 8)
        
    Returns:
        1D array of length grid_size * grid_size (64 values)
    """
    if len(frames_gray) < 2:
        return np.zeros(grid_size * grid_size, dtype=np.float32)

    diff = np.abs(np.diff(frames_gray, axis=0))  # (T-1, H, W)
    t, h, w = diff.shape
    bh, bw = h // grid_size, w // grid_size

    # Reshape into (T-1, grid_size, bh, grid_size, bw)
    trimmed_diff = diff[:, :grid_size * bh, :grid_size * bw]
    blocks = trimmed_diff.reshape(t, grid_size, bh, grid_size, bw)
    # Average over time (axis 0) and within blocks (axis 2, 4)
    block_motion = blocks.mean(axis=(0, 2, 4)).flatten()
    return block_motion.astype(np.float32)


def compute_color_histograms(frames_rgb: np.ndarray) -> np.ndarray:
    """Extract procedural color signatures and RGB color histograms.
    
    Features extracted:
    - 16-bin histograms for R, G, B channels (48 values)
    - Procedural color fractions:
      1. Green drape: (G > 1.2 * R) and (G > 1.1 * B) and (G > 0.25)
      2. Urine yellow: (R > 0.4) and (G > 0.4) and (B < 0.3)
      3. Catheter orange/red: (R > 0.45) and (G < 0.7 * R) and (B < 0.4 * R)
      4. Glove white/light: (R > 0.65) and (G > 0.65) and (B > 0.65)
    
    Returns:
        1D array of 52 values (48 histogram bins + 4 clinical color fractions)
    """
    if len(frames_rgb) == 0:
        return np.zeros(52, dtype=np.float32)

    # Normalize to [0, 1]
    norm_frames = frames_rgb.astype(np.float32) / 255.0
    r = norm_frames[..., 0]
    g = norm_frames[..., 1]
    b = norm_frames[..., 2]

    # Compute 16-bin histograms for each channel
    bins = np.linspace(0.0, 1.0, 17)
    r_hist, _ = np.histogram(r, bins=bins, density=True)
    g_hist, _ = np.histogram(g, bins=bins, density=True)
    b_hist, _ = np.histogram(b, bins=bins, density=True)

    # Clinical color masks
    green_drape = ((g > 1.2 * r) & (g > 1.1 * b) & (g > 0.25)).mean()
    urine_yellow = ((r > 0.4) & (g > 0.4) & (b < 0.3) & (r + g > 1.8 * b)).mean()
    catheter_orange = ((r > 0.45) & (g < 0.75 * r) & (b < 0.45 * r)).mean()
    glove_light = ((r > 0.65) & (g > 0.65) & (b > 0.65)).mean()

    clinical_fractions = np.array([green_drape, urine_yellow, catheter_orange, glove_light], dtype=np.float32)
    hist_features = np.concatenate([r_hist, g_hist, b_hist, clinical_fractions]).astype(np.float32)
    return hist_features


def compute_spatial_summary(frames_gray: np.ndarray, grid_size: int = 8) -> np.ndarray:
    """Compute downsampled 8x8 spatial grayscale summary over the window.
    
    Averages frames across the temporal window and downsamples to 8x8 grid.
    Provides posture, lighting, and procedural field visual context.
    
    Returns:
        1D array of length 64 (8x8 values)
    """
    if len(frames_gray) == 0:
        return np.zeros(grid_size * grid_size, dtype=np.float32)

    mean_frame = frames_gray.mean(axis=0)  # (H, W)
    h, w = mean_frame.shape
    bh, bw = h // grid_size, w // grid_size
    trimmed = mean_frame[:grid_size * bh, :grid_size * bw]
    spatial_summary = trimmed.reshape(grid_size, bh, grid_size, bw).mean(axis=(1, 3)).flatten()
    return spatial_summary.astype(np.float32)


def extract_video_features(frames: np.ndarray) -> np.ndarray:
    """Extract fused video feature vector from RGB frames (T, H, W, 3).
    
    Returns:
        1D array of shape (181,):
        - Frame difference motion energy (1)
        - 8x8 grid block motion (64)
        - Color histograms & clinical color presence (52)
        - Downsampled 8x8 spatial grayscale summary (64)
    """
    if len(frames) == 0:
        return np.zeros(181, dtype=np.float32)

    # Convert RGB to Grayscale using standard ITU-R 601-2 luma transform
    # Y = 0.299 R + 0.587 G + 0.114 B
    frames_gray = (
        0.299 * frames[..., 0] +
        0.587 * frames[..., 1] +
        0.114 * frames[..., 2]
    ).astype(np.float32) / 255.0

    motion_energy = np.array([compute_frame_difference_motion(frames_gray)], dtype=np.float32)
    grid_motion = compute_grid_block_motion(frames_gray, grid_size=8)
    color_hist = compute_color_histograms(frames)
    spatial_summary = compute_spatial_summary(frames_gray, grid_size=8)

    return np.concatenate([motion_energy, grid_motion, color_hist, spatial_summary]).astype(np.float32)


# ============================================================================
# Audio Feature Extraction (Pure NumPy FFT)
# ============================================================================

def hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to Mel scale."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert Mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    n_fft: int = 512,
    sr: int = 16000,
    n_mels: int = 20,
    f_min: float = 50.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Construct triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    # FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    num_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, num_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return filterbank


# Precompute mel filterbank for 16kHz, n_fft=512, 20 mel filters
_DEFAULT_MEL_FB = build_mel_filterbank(n_fft=512, sr=16000, n_mels=20)


def compute_stft(
    audio: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
) -> np.ndarray:
    """Compute Short-Time Fourier Transform magnitude spectrum using NumPy.
    
    Args:
        audio: 1D float array
        n_fft: FFT window size (512 samples = 32ms at 16kHz)
        hop_length: Frame hop (256 samples = 16ms)
        
    Returns:
        spectrogram: 2D array of shape (T_frames, n_fft // 2 + 1)
    """
    if len(audio) < n_fft:
        pad = np.zeros(n_fft - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])

    # Hann window
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1)))
    window = window.astype(np.float32)

    num_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    )
    windowed_frames = frames * window
    # Real FFT
    spec_complex = np.fft.rfft(windowed_frames, n=n_fft, axis=1)
    spec_mag = np.abs(spec_complex).astype(np.float32)
    return spec_mag


def compute_log_mel_features(spec_mag: np.ndarray, mel_fb: np.ndarray = _DEFAULT_MEL_FB) -> np.ndarray:
    """Compute Log-Mel filterbank energies and summarize over time.
    
    Args:
        spec_mag: Magnitude spectrum (T_frames, num_bins)
        mel_fb: Triangular filterbank matrix (n_mels, num_bins)
        
    Returns:
        1D array of length 2 * n_mels (20 means + 20 standard deviations = 40 values)
    """
    power_spec = spec_mag ** 2
    mel_energies = np.dot(power_spec, mel_fb.T)  # (T_frames, n_mels)
    log_mel = np.log(mel_energies + 1e-6)

    mean_mel = log_mel.mean(axis=0)
    std_mel = log_mel.std(axis=0)
    return np.concatenate([mean_mel, std_mel]).astype(np.float32)


def compute_short_time_energy(
    audio: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 256,
    speech_threshold: float = 0.005,
) -> np.ndarray:
    """Compute Short-Time Energy (STE) metrics for speech vs silence detection.
    
    Returns:
        1D array of 3 values: [mean_ste, max_ste, voice_activity_ratio]
    """
    if len(audio) < frame_length:
        return np.zeros(3, dtype=np.float32)

    num_frames = 1 + (len(audio) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, frame_length),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    )
    ste = (frames ** 2).mean(axis=1)

    mean_ste = float(ste.mean())
    max_ste = float(ste.max())
    # Fraction of frames above speech threshold
    vad_ratio = float(np.mean(ste > speech_threshold))
    return np.array([mean_ste, max_ste, vad_ratio], dtype=np.float32)


def compute_spectral_flux(spec_mag: np.ndarray) -> np.ndarray:
    """Compute Spectral Flux to detect transient impulse sounds.
    
    Spectral flux measures the rate of local spectral change between
    consecutive frames: sum(max(0, |X(t, f)| - |X(t-1, f)|)^2).
    Captures clicks, glove snaps, wrapper tears, and syringe pops.
    
    Returns:
        1D array of 3 values: [mean_flux, max_flux, flux_variance]
    """
    if len(spec_mag) < 2:
        return np.zeros(3, dtype=np.float32)

    diff = np.diff(spec_mag, axis=0)
    pos_diff = np.maximum(0.0, diff)
    flux = np.sum(pos_diff ** 2, axis=1)

    mean_flux = float(flux.mean())
    max_flux = float(flux.max())
    var_flux = float(flux.var())
    return np.array([mean_flux, max_flux, var_flux], dtype=np.float32)


def extract_audio_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract fused audio feature vector from 16 kHz audio samples.
    
    Returns:
        1D array of shape (46,):
        - Log-mel mean & std (40)
        - Short-time energy summary: mean, max, voice activity ratio (3)
        - Spectral flux summary: mean, max, variance (3)
    """
    if len(audio) == 0:
        return np.zeros(46, dtype=np.float32)

    spec_mag = compute_stft(audio, n_fft=512, hop_length=256)
    log_mel = compute_log_mel_features(spec_mag)
    ste_feats = compute_short_time_energy(audio, frame_length=512, hop_length=256)
    flux_feats = compute_spectral_flux(spec_mag)

    return np.concatenate([log_mel, ste_feats, flux_feats]).astype(np.float32)


# ============================================================================
# Fused Window Feature Extractor
# ============================================================================

def extract_window_features(
    video_frames: np.ndarray,
    audio_samples: np.ndarray,
    audio_sr: int = 16000,
) -> np.ndarray:
    """Extract and fuse multimodal video and audio features for a sliding window.
    
    Total dimension:
    - Video: 181
    - Audio: 46
    - Total: 227 features
    
    Returns:
        1D array of shape (227,), dtype=float32
    """
    v_feat = extract_video_features(video_frames)
    a_feat = extract_audio_features(audio_samples, sr=audio_sr)
    return np.concatenate([v_feat, a_feat]).astype(np.float32)
