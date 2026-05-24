"""Unit tests for XTTSEngine — mocked, no model weights needed."""
from __future__ import annotations

import io
import numpy as np
import pytest
import soundfile as sf
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wav_bytes(duration: float = 0.5, sr: int = 24000) -> bytes:
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def _dummy_wav_file(path) -> None:
    audio = np.zeros(int(0.5 * 24000), dtype=np.float32)
    sf.write(str(path), audio, 24000)


def _mock_tts_instance(return_samples: int = 12000):
    mock = MagicMock()
    mock.tts.return_value = [0.0] * return_samples
    return mock


@contextmanager
def _patch_settings_dirs(vs_dir: Path, is_dir: Path):
    """Context manager to patch Pydantic @property settings on the Settings class."""
    from configs.settings import Settings
    with (
        patch.object(type(Settings()), "voice_samples_dir",
                     new_callable=PropertyMock, return_value=vs_dir),
        patch.object(type(Settings()), "input_samples_dir",
                     new_callable=PropertyMock, return_value=is_dir),
    ):
        # Patch on the singleton instead
        pass

    # Simpler: patch on the xtts_engine module's imported settings reference
    with (
        patch("backend.engines.xtts_engine.settings") as mock_s,
    ):
        mock_s.voice_samples_dir = vs_dir
        mock_s.input_samples_dir = is_dir
        mock_s.XTTS_MODEL_NAME   = "tts_models/multilingual/multi-dataset/xtts_v2"
        yield mock_s


# ── TestXTTSEngine ────────────────────────────────────────────────────────────

class TestXTTSEngine:

    # ── Engine initialisation ─────────────────────────────────────────────────

    def test_is_ready_true_when_import_succeeds(self):
        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS = engine._model = MagicMock()
            engine._ready = True
            assert engine.is_ready() is True

    def test_is_ready_false_when_import_fails(self):
        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS = engine._model = None
            engine._ready = False
            assert engine.is_ready() is False

    # ── synthesize() error cases ──────────────────────────────────────────────

    def test_raises_runtime_when_not_ready(self):
        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS = engine._model = None
            engine._ready = False
            with pytest.raises(RuntimeError, match="XTTS v2 engine failed"):
                engine.synthesize("hello", voice="clone:test")

    def test_raises_value_error_without_reference_audio(self):
        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS = MagicMock()
            engine._model = None
            engine._ready = True
            with pytest.raises(ValueError, match="reference_audio is required"):
                engine.synthesize("hello", voice="clone:test", reference_audio=None)

    def test_raises_file_not_found_for_missing_reference(self, tmp_path):
        vs_dir = tmp_path / "voice_samples"; vs_dir.mkdir()
        is_dir = tmp_path / "input_samples"; is_dir.mkdir()

        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS = MagicMock()
            engine._model = None
            engine._ready = True

            with _patch_settings_dirs(vs_dir, is_dir):
                with pytest.raises(FileNotFoundError):
                    engine.synthesize("hello", voice="clone:speaker",
                                      reference_audio="nonexistent.wav")

    # ── Language resolution ───────────────────────────────────────────────────

    def test_hindi_language_resolves_to_hi(self, tmp_path):
        self._run_lang_test(tmp_path, language="hi", expected_lang_code="hi")

    def test_japanese_language_resolves_to_ja(self, tmp_path):
        self._run_lang_test(tmp_path, language="ja", expected_lang_code="ja")

    def test_english_us_resolves_to_en(self, tmp_path):
        self._run_lang_test(tmp_path, language="en-us", expected_lang_code="en")

    def test_zh_resolves_to_zh_cn(self, tmp_path):
        self._run_lang_test(tmp_path, language="zh", expected_lang_code="zh-cn")

    def test_pt_br_resolves_to_pt(self, tmp_path):
        self._run_lang_test(tmp_path, language="pt-br", expected_lang_code="pt")

    def _run_lang_test(self, tmp_path, language: str, expected_lang_code: str) -> None:
        vs_dir = tmp_path / "vs"; vs_dir.mkdir()
        is_dir = tmp_path / "is"; is_dir.mkdir()
        ref_file = vs_dir / "ref.wav"
        _dummy_wav_file(ref_file)

        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            mock_model = _mock_tts_instance()
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS   = MagicMock(return_value=mock_model)
            engine._model = mock_model
            engine._ready = True

            with (
                _patch_settings_dirs(vs_dir, is_dir),
                patch("backend.engines.xtts_engine.postprocess",
                      return_value=(np.zeros(100, dtype=np.float32), 24000)),
                patch("backend.engines.xtts_engine.audio_to_bytes",
                      return_value=_wav_bytes()),
            ):
                engine.synthesize("Test text", voice="clone:ref",
                                   language=language, reference_audio="ref.wav")

            call_kwargs = mock_model.tts.call_args
            assert call_kwargs.kwargs.get("language") == expected_lang_code, (
                f"Expected '{expected_lang_code}' but got '{call_kwargs.kwargs.get('language')}'"
            )

    # ── Reference audio resolution ────────────────────────────────────────────

    def test_reference_found_in_voice_samples_dir(self, tmp_path):
        vs_dir = tmp_path / "vs"; vs_dir.mkdir()
        is_dir = tmp_path / "is"; is_dir.mkdir()
        _dummy_wav_file(vs_dir / "speaker.wav")

        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            mock_model = _mock_tts_instance()
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS   = MagicMock(return_value=mock_model)
            engine._model = mock_model
            engine._ready = True

            with (
                _patch_settings_dirs(vs_dir, is_dir),
                patch("backend.engines.xtts_engine.postprocess",
                      return_value=(np.zeros(100, dtype=np.float32), 24000)),
                patch("backend.engines.xtts_engine.audio_to_bytes",
                      return_value=_wav_bytes()),
            ):
                engine.synthesize("hello", voice="clone:speaker",
                                   language="en", reference_audio="speaker.wav")

        mock_model.tts.assert_called_once()

    def test_reference_fallback_to_input_samples_dir(self, tmp_path):
        vs_dir = tmp_path / "vs"; vs_dir.mkdir()
        is_dir = tmp_path / "is"; is_dir.mkdir()
        _dummy_wav_file(is_dir / "hindi_ref.wav")  # only in input_samples

        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            mock_model = _mock_tts_instance()
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS   = MagicMock(return_value=mock_model)
            engine._model = mock_model
            engine._ready = True

            with (
                _patch_settings_dirs(vs_dir, is_dir),
                patch("backend.engines.xtts_engine.postprocess",
                      return_value=(np.zeros(100, dtype=np.float32), 24000)),
                patch("backend.engines.xtts_engine.audio_to_bytes",
                      return_value=_wav_bytes()),
            ):
                # Should NOT raise — file exists in input_samples_dir
                engine.synthesize("hello", voice="clone:hindi_ref",
                                   language="hi", reference_audio="hindi_ref.wav")

        mock_model.tts.assert_called_once()

    # ── list_voices() ─────────────────────────────────────────────────────────

    def test_list_voices_returns_clone_entries(self, tmp_path):
        vs_dir = tmp_path / "voice_samples"; vs_dir.mkdir()
        _dummy_wav_file(vs_dir / "hindi_speaker.wav")
        _dummy_wav_file(vs_dir / "jp_speaker.wav")
        (vs_dir / "ignored.txt").write_text("not audio")

        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS = engine._model = None
            engine._ready = True

            with patch("backend.engines.xtts_engine.settings") as ms:
                ms.voice_samples_dir = vs_dir
                voices = engine.list_voices()

        ids = [v["id"] for v in voices]
        assert "clone:hindi_speaker" in ids
        assert "clone:jp_speaker" in ids
        assert all(v["engine"] == "xtts-v2" for v in voices)
        assert not any("ignored" in v["id"] for v in voices)

    def test_list_voices_empty_dir(self, tmp_path):
        vs_dir = tmp_path / "voice_samples"; vs_dir.mkdir()

        with patch("backend.engines.xtts_engine.XTTSEngine._load"):
            from backend.engines.xtts_engine import XTTSEngine
            engine = XTTSEngine.__new__(XTTSEngine)
            engine._TTS = engine._model = None
            engine._ready = True

            with patch("backend.engines.xtts_engine.settings") as ms:
                ms.voice_samples_dir = vs_dir
                voices = engine.list_voices()

        assert voices == []
