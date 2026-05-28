"""Unit tests for audio utility functions."""
import numpy as np
import pytest

from backend.utils.audio import resample, normalize_loudness, trim_silence, postprocess


def _sine(freq=440, duration=1.0, sr=44100) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def test_resample_changes_length():
    audio = _sine(sr=44100)
    resampled = resample(audio, src_rate=44100, target_rate=24000)
    expected_len = int(len(audio) * 24000 / 44100)
    assert abs(len(resampled) - expected_len) <= 10


def test_resample_no_op_same_rate():
    audio = _sine(sr=24000)
    result = resample(audio, src_rate=24000, target_rate=24000)
    np.testing.assert_array_equal(result, audio)


def test_normalize_loudness_peak():
    audio = np.array([0.1, -0.5, 0.3, -0.2], dtype=np.float32)
    normalized = normalize_loudness(audio, target_db=-20.0)
    target_amp = 10 ** (-20.0 / 20.0)
    assert abs(np.abs(normalized).max() - target_amp) < 1e-5


def test_normalize_silence_passthrough():
    audio = np.zeros(100, dtype=np.float32)
    result = normalize_loudness(audio)
    np.testing.assert_array_equal(result, audio)


def test_trim_silence_removes_edges():
    silence = np.zeros(1000, dtype=np.float32)
    signal = np.ones(500, dtype=np.float32) * 0.5
    audio = np.concatenate([silence, signal, silence])
    trimmed = trim_silence(audio, threshold_db=-50.0)
    assert len(trimmed) < len(audio)
    assert len(trimmed) >= 500


def test_postprocess_output_sample_rate():
    audio = _sine(sr=44100, duration=0.5)
    processed, sr = postprocess(audio, src_rate=44100)
    from configs.settings import settings
    assert sr == settings.TARGET_SAMPLE_RATE
