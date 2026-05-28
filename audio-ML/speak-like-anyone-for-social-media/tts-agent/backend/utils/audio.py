import io
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

from configs.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def resample(audio: np.ndarray, src_rate: int, target_rate: int | None = None) -> np.ndarray:
    """Resample audio array to target sample rate."""
    rate = target_rate or settings.TARGET_SAMPLE_RATE
    if src_rate == rate:
        return audio
    tensor = torch.from_numpy(audio).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    resampler = T.Resample(orig_freq=src_rate, new_freq=rate)
    resampled = resampler(tensor).squeeze(0).numpy()
    logger.debug("Resampled %d Hz → %d Hz", src_rate, rate)
    return resampled


def normalize_loudness(audio: np.ndarray, target_db: float | None = None) -> np.ndarray:
    """Peak-normalize audio to target dBFS."""
    target = target_db or settings.TARGET_LOUDNESS_DB
    peak = np.abs(audio).max()
    if peak < 1e-9:
        return audio
    target_amplitude = 10 ** (target / 20.0)
    return (audio / peak) * target_amplitude


def trim_silence(audio: np.ndarray, threshold_db: float = -50.0) -> np.ndarray:
    """Remove leading/trailing silence below threshold."""
    threshold_amp = 10 ** (threshold_db / 20.0)
    mask = np.abs(audio) > threshold_amp
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return audio
    return audio[indices[0] : indices[-1] + 1]


def audio_to_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


def save_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    logger.info("Saved WAV → %s", path)


def load_audio(path: Path, target_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio file and optionally resample."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono
    if target_rate and sr != target_rate:
        audio = resample(audio, sr, target_rate)
        sr = target_rate
    return audio, sr


def postprocess(audio: np.ndarray, src_rate: int) -> tuple[np.ndarray, int]:
    """Standard pipeline: resample → normalize → trim silence."""
    target = settings.TARGET_SAMPLE_RATE
    audio = resample(audio, src_rate, target)
    audio = normalize_loudness(audio)
    audio = trim_silence(audio)
    return audio, target
