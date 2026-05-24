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


class TestXTTSEngine:
    def test_synthesize_raises_without_reference(self):
        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS = MagicMock()
            engine._model = None
            engine._ready = True
            with pytest.raises(ValueError, match="reference_audio is required"):
                engine.synthesize("hello", voice="clone:speaker")

    def test_synthesize_raises_when_not_ready(self):
        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS = None
            engine._model = None
            engine._ready = False
            with pytest.raises(RuntimeError, match="XTTS v2 engine failed"):
                engine.synthesize("hello", voice="clone:speaker",
                                  reference_audio="speaker.wav")

    def test_list_voices_from_samples_dir(self, tmp_path):
        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS = engine._model = None
            engine._ready = True

            (tmp_path / "speaker1.wav").touch()
            (tmp_path / "speaker2.wav").touch()

            with patch("backend.engines.xtts_engine.settings") as ms:
                ms.voice_samples_dir = tmp_path
                voices = engine.list_voices()

        ids = [v["id"] for v in voices]
        assert "clone:speaker1" in ids
        assert "clone:speaker2" in ids
        assert all(v["engine"] == "xtts-v2" for v in voices)
