"""Engine unit tests using mocks — no model weights needed."""
import io
import numpy as np
import pytest
import soundfile as sf
from unittest.mock import MagicMock, patch


def _wav_bytes(duration=0.5, sr=24000) -> bytes:
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class TestKokoroEngine:
    def test_list_voices_returns_english_and_hindi(self):
        with patch("backend.engines.kokoro_engine.KokoroEngine._load"):
            from backend.engines.kokoro_engine import KokoroEngine, _VOICES
            engine = KokoroEngine.__new__(KokoroEngine)
            engine._pipelines = {}
            engine._ready = True
            voices = engine.list_voices()
            langs = {v["language"] for v in voices}
            assert "en-us" in langs
            assert "hi" in langs

    def test_synthesize_raises_on_unknown_voice(self):
        with patch("backend.engines.kokoro_engine.KokoroEngine._load"):
            from backend.engines.kokoro_engine import KokoroEngine
            engine = KokoroEngine.__new__(KokoroEngine)
            engine._pipelines = {}
            engine._ready = True
            engine._KPipeline = MagicMock()
            with pytest.raises(ValueError, match="Unknown Kokoro voice"):
                engine.synthesize("hello", voice="not_a_real_voice")

    def test_synthesize_raises_when_not_ready(self):
        with patch("backend.engines.kokoro_engine.KokoroEngine._load"):
            from backend.engines.kokoro_engine import KokoroEngine
            engine = KokoroEngine.__new__(KokoroEngine)
            engine._pipelines = {}
            engine._ready = False
            with pytest.raises(RuntimeError):
                engine.synthesize("hello", voice="af_bella")


class TestF5Engine:
    def test_synthesize_raises_without_reference(self):
        with patch("backend.engines.f5_engine.F5Engine._load"):
            from backend.engines.f5_engine import F5Engine
            engine = F5Engine.__new__(F5Engine)
            engine._model = None
            engine._ready = True
            engine._F5TTS = MagicMock()
            with pytest.raises(ValueError, match="reference_audio"):
                engine.synthesize("hello", voice="clone:speaker")

    def test_list_voices_from_samples_dir(self, tmp_path):
        with patch("backend.engines.f5_engine.F5Engine._load"):
            from backend.engines.f5_engine import F5Engine
            from configs.settings import settings
            engine = F5Engine.__new__(F5Engine)
            engine._model = None
            engine._ready = True

            # Create dummy wav files
            (tmp_path / "speaker1.wav").touch()
            (tmp_path / "speaker2.wav").touch()

            with patch.object(settings, "voice_samples_dir", tmp_path):
                voices = engine.list_voices()
            ids = [v["id"] for v in voices]
            assert "clone:speaker1" in ids
            assert "clone:speaker2" in ids
