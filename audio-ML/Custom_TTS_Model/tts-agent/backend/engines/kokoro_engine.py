"""Kokoro-82M TTS engine wrapper (Apache 2.0 licensed — commercial-safe)."""
import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.engines.base import TTSEngine
from backend.utils.audio import postprocess, audio_to_bytes
from backend.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)

_KOKORO_REPO_ID = "hexgrad/Kokoro-82M"


def _configure_espeak() -> None:
    """
    Point espeak-ng at the bundled data directory shipped by espeakng_loader.

    Root cause of 'Error processing file .../espeakng_loader//phontab':
      espeak_ng_InitializePath() has an internal 160-char snprintf buffer.
      When the full venv path exceeds that limit the string is silently
      truncated, both directory checks fail, and the C library falls through
      to its compiled-in CI build path ('.../runner/work/...') which does
      not exist on the user's machine.

    Fix: if the real data path is longer than 140 chars, create a short
    symlink under /tmp and point both ESPEAK_DATA_PATH and phonemizer at
    that symlink instead.  The symlink is idempotent and cheap.
    """
    try:
        import espeakng_loader
        data_path: str = espeakng_loader.get_data_path()
    except Exception as exc:
        logger.warning("espeakng_loader not available: %s", exc)
        return

    # ------------------------------------------------------------------
    # Workaround: shorten the path if it would overflow the C buffer
    # ------------------------------------------------------------------
    if len(data_path) > 140:
        short = Path("/tmp/kokoro_espeak_data")
        try:
            if short.is_symlink() and str(short.resolve()) != str(Path(data_path).resolve()):
                short.unlink()
            if not short.exists():
                short.symlink_to(data_path)
            data_path = str(short)
            logger.debug("espeak data path aliased to %s", data_path)
        except Exception as exc:
            logger.warning("Could not create espeak data symlink: %s", exc)

    # Belt-and-braces: set the env var AND the phonemizer class variable
    os.environ["ESPEAK_DATA_PATH"] = data_path
    os.environ["PHONEMIZER_ESPEAK_DATA_PATH"] = data_path
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_data_path(data_path)
        logger.debug("EspeakWrapper.set_data_path → %s", data_path)
    except Exception as exc:
        logger.warning("Could not set EspeakWrapper data path: %s", exc)


# Configure espeak at import time — must happen before KPipeline is instantiated
_configure_espeak()

# Voice catalog: id → (display name, lang_code for KPipeline)
_VOICES: dict[str, dict] = {
    # American English
    "af_bella":   {"name": "Bella (EN Female)",   "language": "en-us", "lang_code": "a"},
    "af_nicole":  {"name": "Nicole (EN Female)",  "language": "en-us", "lang_code": "a"},
    "af_sarah":   {"name": "Sarah (EN Female)",   "language": "en-us", "lang_code": "a"},
    "af_sky":     {"name": "Sky (EN Female)",     "language": "en-us", "lang_code": "a"},
    "am_adam":    {"name": "Adam (EN Male)",      "language": "en-us", "lang_code": "a"},
    "am_michael": {"name": "Michael (EN Male)",   "language": "en-us", "lang_code": "a"},
    # British English
    "bf_emma":    {"name": "Emma (EN-GB Female)", "language": "en-gb", "lang_code": "b"},
    "bf_isabella":{"name": "Isabella (EN-GB Female)", "language": "en-gb", "lang_code": "b"},
    "bm_george":  {"name": "George (EN-GB Male)", "language": "en-gb", "lang_code": "b"},
    "bm_lewis":   {"name": "Lewis (EN-GB Male)",  "language": "en-gb", "lang_code": "b"},
    # Hindi
    "hf_alpha":   {"name": "Alpha (HI Female)",   "language": "hi",    "lang_code": "h"},
    "hf_beta":    {"name": "Beta (HI Female)",    "language": "hi",    "lang_code": "h"},
    "hm_omega":   {"name": "Omega (HI Male)",     "language": "hi",    "lang_code": "h"},
}


class KokoroEngine(TTSEngine):
    """Lazily loads KPipeline instances keyed by lang_code."""

    def __init__(self) -> None:
        self._pipelines: dict[str, object] = {}
        self._ready = False
        self._load()

    def _load(self) -> None:
        try:
            from kokoro import KPipeline  # noqa: F401 — validate import at startup
            self._KPipeline = KPipeline
            self._ready = True
            logger.info("Kokoro engine initialized (models load on first synthesis call)")
        except ImportError as exc:
            logger.error("Kokoro package not installed: %s", exc)

    def _get_pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            logger.info("Loading Kokoro pipeline for lang_code='%s'", lang_code)
            self._pipelines[lang_code] = self._KPipeline(
                lang_code=lang_code,
                repo_id=_KOKORO_REPO_ID,
            )
        return self._pipelines[lang_code]

    def synthesize(
        self,
        text: str,
        voice: str,
        language: str = "en-us",
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        if not self._ready:
            raise RuntimeError("Kokoro engine failed to initialize. Check installation.")

        voice_meta = _VOICES.get(voice)
        if voice_meta is None:
            raise ValueError(f"Unknown Kokoro voice: {voice}")

        lang_code = voice_meta["lang_code"]
        pipeline = self._get_pipeline(lang_code)

        chunks: list[np.ndarray] = []
        # KPipeline returns a generator of (graphemes, phonemes, audio_array) tuples
        for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
            if audio is not None and len(audio) > 0:
                chunks.append(np.asarray(audio, dtype=np.float32))

        if not chunks:
            raise RuntimeError("Kokoro produced no audio output.")

        combined = np.concatenate(chunks)
        processed, sr = postprocess(combined, src_rate=24000)
        return audio_to_bytes(processed, sr)

    def list_voices(self) -> list[dict]:
        return [
            {"id": vid, "name": meta["name"], "language": meta["language"], "engine": "kokoro"}
            for vid, meta in _VOICES.items()
        ]

    def is_ready(self) -> bool:
        return self._ready
