"""
Broadcast-grade audio post-processing pipeline.

Applies in order:
  1. Resample to 44.1 kHz
  2. Noise reduction (spectral subtraction)
  3. High-pass filter @ 80 Hz
  4. Presence EQ shelf @ 3 kHz
  5. EBU R128 loudness normalisation
  6. True-peak limiter
  7. Mono → stereo
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, resample_poly
from math import gcd

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Broadcast loudness profiles (integrated LUFS target)
LOUDNESS_PROFILES = {
    "broadcast":  -23.0,   # EBU R128 / EBU TECH 3343
    "streaming":  -14.0,   # Spotify / Apple Music
    "podcast":    -16.0,   # Podtrac / Buzzsprout recommendation
    "film":       -24.0,   # SMPTE RP 2095-1
}

TARGET_SAMPLE_RATE = 44_100
TRUE_PEAK_LIMIT_DB = -1.0


# ── Resampler ─────────────────────────────────────────────────────────────────

def resample_to_44k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """High-quality rational resampling to 44100 Hz using polyphase filter."""
    if src_rate == TARGET_SAMPLE_RATE:
        return audio.copy()
    divisor = gcd(src_rate, TARGET_SAMPLE_RATE)
    up   = TARGET_SAMPLE_RATE // divisor
    down = src_rate // divisor
    resampled = resample_poly(audio, up, down)
    logger.debug("Resampled %d Hz → %d Hz  (up=%d down=%d)", src_rate, TARGET_SAMPLE_RATE, up, down)
    return resampled.astype(np.float32)


# ── Noise reduction ───────────────────────────────────────────────────────────

def reduce_noise(audio: np.ndarray, sr: int, prop_decrease: float = 0.75) -> np.ndarray:
    """Spectral subtraction noise reduction via noisereduce."""
    try:
        import noisereduce as nr
        reduced = nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease, stationary=False)
        return reduced.astype(np.float32)
    except ImportError:
        logger.warning("noisereduce not installed — skipping noise reduction. pip install noisereduce")
        return audio


# ── Parametric EQ ─────────────────────────────────────────────────────────────

def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0, order: int = 2) -> np.ndarray:
    """Remove low-frequency rumble below cutoff."""
    nyq = sr / 2.0
    sos = butter(order, cutoff_hz / nyq, btype="high", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def presence_shelf(audio: np.ndarray, sr: int, freq_hz: float = 3000.0, gain_db: float = 2.0) -> np.ndarray:
    """High-shelf boost for narration clarity using a simple shelving filter."""
    nyq = sr / 2.0
    norm_freq = freq_hz / nyq
    # First-order high-shelf approximation via biquad via sos
    K = np.tan(np.pi * norm_freq)
    V0 = 10 ** (gain_db / 20.0)
    b0 = (V0 + V0 * K) / (1 + K)
    b1 = (V0 * K - V0) / (1 + K)
    a1 = (K - 1) / (1 + K)
    sos = np.array([[b0, b1, 0.0, 1.0, a1, 0.0]])
    return sosfilt(sos, audio).astype(np.float32)


# ── EBU R128 loudness normalisation ──────────────────────────────────────────

def normalise_loudness(
    audio: np.ndarray,
    sr: int,
    profile: str = "broadcast",
) -> np.ndarray:
    """Measure integrated loudness (pyloudnorm) and apply gain to hit target LUFS."""
    target_lufs = LOUDNESS_PROFILES.get(profile, LOUDNESS_PROFILES["broadcast"])
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr, block_size=0.400)
        # pyloudnorm needs (samples, channels)
        audio_2d = audio[:, np.newaxis] if audio.ndim == 1 else audio
        measured = meter.integrated_loudness(audio_2d)
        if not np.isfinite(measured):
            logger.warning("Loudness measurement returned -inf — audio may be silence. Skipping gain.")
            return audio
        gain_db = target_lufs - measured
        gain_linear = 10 ** (gain_db / 20.0)
        logger.info("Loudness: %.1f LUFS → target %.1f LUFS (gain %+.1f dB)", measured, target_lufs, gain_db)
        return (audio * gain_linear).astype(np.float32)
    except ImportError:
        logger.warning("pyloudnorm not installed — skipping loudness normalisation. pip install pyloudnorm")
        # Fallback: peak normalise to -1 dBFS
        peak = np.abs(audio).max()
        if peak > 1e-9:
            audio = audio / peak * (10 ** (-1.0 / 20.0))
        return audio.astype(np.float32)


# ── True-peak limiter ─────────────────────────────────────────────────────────

def true_peak_limit(audio: np.ndarray, limit_db: float = TRUE_PEAK_LIMIT_DB) -> np.ndarray:
    """Hard clip to prevent inter-sample peaks on D/A conversion."""
    limit_amp = 10 ** (limit_db / 20.0)
    return np.clip(audio, -limit_amp, limit_amp).astype(np.float32)


# ── Stereo conversion ─────────────────────────────────────────────────────────

def mono_to_stereo(audio: np.ndarray) -> np.ndarray:
    """Duplicate mono channel to stereo (L+R identical)."""
    if audio.ndim == 2 and audio.shape[1] == 2:
        return audio
    mono = audio.squeeze() if audio.ndim > 1 else audio
    return np.stack([mono, mono], axis=1).astype(np.float32)


# ── Full broadcast pipeline ───────────────────────────────────────────────────

def broadcast_postprocess(
    audio: np.ndarray,
    src_rate: int,
    *,
    loudness_profile: str = "broadcast",
    noise_reduce: bool = True,
    eq: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Apply the full broadcast post-processing chain.

    Returns (stereo_float32_at_44100, 44100).
    """
    # 1. Resample
    audio = resample_to_44k(audio, src_rate)
    sr = TARGET_SAMPLE_RATE

    # 2. Noise reduction
    if noise_reduce:
        audio = reduce_noise(audio, sr)

    # 3. High-pass filter
    if eq:
        audio = highpass_filter(audio, sr, cutoff_hz=80.0)

    # 4. Presence shelf
    if eq:
        audio = presence_shelf(audio, sr, freq_hz=3000.0, gain_db=2.0)

    # 5. Loudness normalisation (EBU R128)
    audio = normalise_loudness(audio, sr, profile=loudness_profile)

    # 6. True-peak limit
    audio = true_peak_limit(audio)

    # 7. Mono → stereo
    audio = mono_to_stereo(audio)

    return audio, sr
