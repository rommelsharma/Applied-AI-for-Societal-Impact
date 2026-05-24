"""XTTS v2 engine — multilingual zero-shot voice cloning (Coqui TTS).

Model
-----
tts_models/multilingual/multi-dataset/xtts_v2

Auto-downloads ~1.8 GB on first synthesis call.  Pre-download with:
    python scripts/download_models.py --xtts

Supported languages (17)
------------------------
en  es  fr  de  it  pt  pl  tr  ru  nl  cs  ar  zh-cn  hu  ko  ja  hi

Voice cloning
-------------
Provide a 5-15 second WAV clip of a single speaker as reference_audio.
The engine searches voice_samples/ then input_samples/ for the file.
Reference transcript (reference_text) is accepted but not used — XTTS v2
performs its own speaker embedding extraction directly from the audio waveform.

Speed
-----
XTTS v2 does not expose an inference-time speed multiplier.  The speed
parameter is accepted (for API compatibility) but not forwarded to the model.
Post-synthesis resampling for speed control can be added as a future enhancement.

License
-------
Coqui Public Model License v1.0 — commercial use allowed for revenues under
$1 M/yr.  The TTS library code itself is MIT.
See: https://coqui.ai/cpml
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from backend.engines.base import TTSEngine
from backend.utils.audio import postprocess, audio_to_bytes
from backend.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)

# ── Language map: BCP-47 / ISO 639-1 → XTTS language code ────────────────────
# XTTS v2 accepts lowercase 2-letter codes or "zh-cn" for Mandarin.
_LANG_MAP: dict[str, str] = {
    # English
    "en": "en", "en-us": "en", "en-gb": "en",
    # Hindi — added to XTTS v2 alongside the 16 primary training languages
    "hi": "hi",
    # Japanese
    "ja": "ja",
    # Other supported languages
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt", "pt-br": "pt",
    "pl": "pl",
    "tr": "tr",
    "ru": "ru",
    "nl": "nl",
    "cs": "cs",
    "ar": "ar",
    "zh": "zh-cn", "zh-cn": "zh-cn",
    "hu": "hu",
    "ko": "ko",
}

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


class XTTSEngine(TTSEngine):
    """Coqui XTTS v2 — single-model multilingual zero-shot voice cloning."""

    def __init__(self) -> None:
        self._TTS = None     # TTS class (imported lazily to avoid heavy import at startup)
        self._model = None   # loaded TTS instance (lazy, created on first synthesis)
        self._ready = False
        self._load()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Import Coqui TTS library.  Keeps server startup fast even if TTS is absent."""
        try:
            from TTS.api import TTS  # noqa: F401 — validate import at startup
            self._TTS = TTS
            self._ready = True
            logger.info("XTTS v2 engine initialized (model loads on first synthesis call)")
        except ImportError as exc:
            logger.error(
                "Coqui TTS package not installed: %s  —  "
                "Install with: pip install TTS>=0.22.0",
                exc,
            )

    def _get_model(self):
        """Lazy-load the XTTS v2 model on the first synthesis call."""
        if self._model is None:
            from backend.utils.device import DEVICE
            logger.info(
                "[XTTS] Loading model '%s' on device '%s' (first call triggers ~1.8 GB download if not cached)…",
                settings.XTTS_MODEL_NAME, DEVICE,
            )
            try:
                self._model = self._TTS(settings.XTTS_MODEL_NAME).to(DEVICE)
            except Exception as exc:
                # MPS sometimes rejects XTTS — fall back to CPU
                if "mps" in str(DEVICE).lower():
                    logger.warning("[XTTS] MPS failed (%s) — retrying on CPU", exc)
                    self._model = self._TTS(settings.XTTS_MODEL_NAME).to("cpu")
                else:
                    raise
            logger.info("[XTTS] Model loaded successfully.")
        return self._model

    # ── Reference audio resolution ────────────────────────────────────────────

    def _resolve_reference(self, reference_audio: str) -> Path:
        """Find reference_audio in voice_samples/ then input_samples/.

        This allows the test suite to use files from input_samples/ without
        requiring the user to upload them to voice_samples/ first.
        """
        for search_dir in (settings.voice_samples_dir, settings.input_samples_dir):
            p = search_dir / Path(reference_audio).name  # strip any path traversal
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Reference audio '{reference_audio}' not found in "
            f"voice_samples/ or input_samples/. "
            f"Upload the file or add it to input_samples/."
        )

    # ── TTSEngine contract ────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice: str,
        language: str = "en-us",
        speed: float = 1.0,
        reference_audio: str | None = None,
        reference_text: str | None = None,
        **kwargs,
    ) -> bytes:
        """
        Synthesize text in the cloned voice described by reference_audio.

        Parameters
        ----------
        text : str
            The text to synthesize.
        voice : str
            Voice ID in the format ``clone:<stem>`` (used for logging only).
        language : str
            BCP-47 tag or ISO 639-1 code for the *output* language.
        speed : float
            Accepted for API compatibility; not forwarded to XTTS v2.
        reference_audio : str
            Filename of the reference WAV (searched in voice_samples/ then input_samples/).
        reference_text : str | None
            Optional transcript of the reference clip (not used by XTTS v2).
        """
        if not self._ready:
            raise RuntimeError(
                "XTTS v2 engine failed to initialize. "
                "Run: pip install TTS>=0.22.0"
            )
        if not reference_audio:
            raise ValueError(
                "reference_audio is required for XTTS v2 voice cloning. "
                "Upload a WAV clip or place a file in input_samples/."
            )

        ref_path = self._resolve_reference(reference_audio)

        # Normalise language tag to XTTS code (e.g. "en-us" → "en")
        lang_lower = language.lower()
        lang_code = _LANG_MAP.get(lang_lower) or _LANG_MAP.get(lang_lower.split("-")[0], "en")

        logger.info(
            "[XTTS] request | voice=%s | lang=%s→%s | chars=%d | ref=%s",
            voice, language, lang_code, len(text), ref_path.name,
        )

        model = self._get_model()

        # XTTS v2 returns a list of float32 audio samples at 24 kHz
        wav: list[float] = model.tts(
            text=text,
            speaker_wav=str(ref_path),
            language=lang_code,
        )

        audio = np.array(wav, dtype=np.float32)
        processed, sr = postprocess(audio, src_rate=24000)
        duration = len(processed) / sr if len(processed) > 0 else 0.0

        logger.info(
            "[XTTS] output | voice=%s | lang=%s | duration=%.2fs | sample_rate=%d",
            voice, lang_code, duration, sr,
        )

        return audio_to_bytes(processed, sr)

    def list_voices(self) -> list[dict]:
        """Return one entry per audio file in voice_samples/."""
        voices: list[dict] = []
        for p in sorted(settings.voice_samples_dir.glob("*")):
            if p.suffix.lower() in _AUDIO_EXTENSIONS:
                voices.append({
                    "id": f"clone:{p.stem}",
                    "name": f"{p.stem} (Cloned)",
                    "language": "any",
                    "engine": "xtts-v2",
                    "reference_file": p.name,
                })
        return voices

    def is_ready(self) -> bool:
        return self._ready
