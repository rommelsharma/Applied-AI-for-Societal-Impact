"""Unit tests for the broadcast post-processing pipeline."""
import numpy as np
import pytest
from unittest.mock import patch


def _sine(freq=440, duration=1.0, sr=24000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


class TestResampleTo44k:
    def test_output_sample_rate(self):
        from backend.utils.broadcast import resample_to_44k
        audio = _sine(sr=24000)
        out = resample_to_44k(audio, src_rate=24000)
        expected_len = int(len(audio) * 44100 / 24000)
        assert abs(len(out) - expected_len) <= 20

    def test_no_op_when_already_44k(self):
        from backend.utils.broadcast import resample_to_44k
        audio = _sine(sr=44100)
        out = resample_to_44k(audio, src_rate=44100)
        np.testing.assert_array_equal(out, audio)

    def test_from_22050(self):
        from backend.utils.broadcast import resample_to_44k
        audio = _sine(sr=22050)
        out = resample_to_44k(audio, src_rate=22050)
        expected_len = int(len(audio) * 44100 / 22050)
        assert abs(len(out) - expected_len) <= 5


class TestHighpassFilter:
    def test_attenuates_low_frequency(self):
        from backend.utils.broadcast import highpass_filter
        sr = 44100
        # Generate 20 Hz tone (should be heavily attenuated)
        t = np.linspace(0, 1.0, sr, endpoint=False)
        low_freq = np.sin(2 * np.pi * 20 * t).astype(np.float32)
        filtered = highpass_filter(low_freq, sr, cutoff_hz=80.0)
        assert np.abs(filtered).max() < np.abs(low_freq).max() * 0.2

    def test_passes_high_frequency(self):
        from backend.utils.broadcast import highpass_filter
        sr = 44100
        # 1 kHz tone (well above cutoff — should pass through)
        t = np.linspace(0, 1.0, sr, endpoint=False)
        high_freq = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        filtered = highpass_filter(high_freq, sr, cutoff_hz=80.0)
        # RMS ratio should be > 0.9 (< 1 dB attenuation)
        rms_in = np.sqrt(np.mean(high_freq ** 2))
        rms_out = np.sqrt(np.mean(filtered ** 2))
        assert rms_out / rms_in > 0.9


class TestTruePeakLimit:
    def test_clips_above_limit(self):
        from backend.utils.broadcast import true_peak_limit
        audio = np.array([0.5, 0.95, 1.2, -1.5, 0.3], dtype=np.float32)
        limited = true_peak_limit(audio, limit_db=-1.0)
        limit_amp = 10 ** (-1.0 / 20.0)
        assert np.abs(limited).max() <= limit_amp + 1e-6

    def test_preserves_below_limit(self):
        from backend.utils.broadcast import true_peak_limit
        audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        limited = true_peak_limit(audio)
        np.testing.assert_array_almost_equal(limited, audio)


class TestMonoToStereo:
    def test_mono_becomes_stereo(self):
        from backend.utils.broadcast import mono_to_stereo
        audio = _sine(sr=44100)
        stereo = mono_to_stereo(audio)
        assert stereo.ndim == 2
        assert stereo.shape[1] == 2

    def test_channels_identical(self):
        from backend.utils.broadcast import mono_to_stereo
        audio = _sine(sr=44100)
        stereo = mono_to_stereo(audio)
        np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1])

    def test_stereo_input_passthrough(self):
        from backend.utils.broadcast import mono_to_stereo
        audio = _sine(sr=44100)
        stereo_in = np.stack([audio, audio], axis=1)
        result = mono_to_stereo(stereo_in)
        assert result.shape == stereo_in.shape


class TestLoudnessNormalisation:
    def test_gain_applied(self):
        """Loudness normalisation should change the amplitude."""
        pytest.importorskip("pyloudnorm")
        from backend.utils.broadcast import normalise_loudness
        audio = _sine(freq=440, duration=2.0, sr=44100)
        result = normalise_loudness(audio, sr=44100, profile="broadcast")
        assert not np.allclose(result, audio, atol=1e-3)

    def test_silence_passthrough(self):
        from backend.utils.broadcast import normalise_loudness
        audio = np.zeros(44100, dtype=np.float32)
        result = normalise_loudness(audio, sr=44100)
        np.testing.assert_array_equal(result, audio)


class TestBroadcastPipeline:
    def test_output_is_stereo_44k(self):
        from backend.utils.broadcast import broadcast_postprocess
        audio = _sine(sr=24000, duration=1.0)
        out, sr = broadcast_postprocess(audio, src_rate=24000, noise_reduce=False)
        assert sr == 44100
        assert out.ndim == 2
        assert out.shape[1] == 2
