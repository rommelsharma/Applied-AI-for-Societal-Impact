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

import threading
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
        self._TTS = None      # TTS class (imported lazily to avoid heavy import at startup)
        self._model = None    # loaded TTS instance (lazy, created on first synthesis)
        self._ready = False
        self._model_lock = threading.Lock()      # prevents double-load on concurrent first requests
        self._inference_lock = threading.Lock()  # XTTS v2 model.tts() is not thread-safe
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

    @staticmethod
    def _patch_torchaudio() -> None:
        """Patch torchaudio.load to use soundfile when TorchCodec is not installed.

        torchaudio ≥2.9 replaced its audio backend with TorchCodec.  When that
        package is absent (e.g. macOS MPS environments), torchaudio.load raises
        ``ImportError: TorchCodec is required``.  We intercept the import and
        substitute a soundfile-based loader that returns the same (tensor, sr)
        tuple so Coqui TTS works transparently.
        """
        try:
            from torchcodec.decoders import AudioDecoder  # noqa: F401 — already installed, nothing to do
            return
        except ImportError:
            pass  # torchcodec absent — apply patch below

        try:
            import torchaudio
            import torch
            import soundfile as sf  # noqa: F401 — validate availability

            def _sf_load(
                uri,
                frame_offset: int = 0,
                num_frames: int = -1,
                normalize: bool = True,
                channels_first: bool = True,
                format=None,
                buffer_size: int = 4096,
                backend=None,
            ):
                import soundfile as _sf
                import numpy as _np
                import torch as _torch

                data, sr = _sf.read(str(uri), dtype="float32", always_2d=True)
                # data: [time, channels] → tensor: [channels, time]
                tensor = _torch.from_numpy(_np.ascontiguousarray(data.T))
                if not channels_first:
                    tensor = tensor.T
                if frame_offset:
                    tensor = tensor[..., frame_offset:]
                if num_frames > 0:
                    tensor = tensor[..., :num_frames]
                return tensor, sr

            torchaudio.load = _sf_load
            logger.info(
                "[XTTS] torchaudio.load patched — using soundfile backend "
                "(TorchCodec not installed; install torchcodec to use native backend)"
            )
        except Exception as exc:
            logger.warning("[XTTS] Could not patch torchaudio.load: %s", exc)

    def _get_model(self):
        """Lazy-load the XTTS v2 model on the first synthesis call.

        A threading.Lock guards the initialisation block so that concurrent
        synthesis requests (e.g. "Run All" firing three tests at once) do not
        each attempt to load the 1.8 GB model simultaneously.
        """
        if self._model is None:
            with self._model_lock:
                # Double-checked locking: re-test inside the lock in case
                # another thread loaded the model while we were waiting.
                if self._model is None:
                    import torch
                    from backend.utils.device import DEVICE

                    # torchaudio ≥2.9 requires TorchCodec for torchaudio.load().
                    # Patch it to use soundfile when TorchCodec is absent.
                    self._patch_torchaudio()

                    # PyTorch 2.6+ defaults torch.load to weights_only=True, which
                    # blocks Coqui TTS checkpoint classes.  Register them as trusted
                    # globals so model loading works without disabling safety checks.
                    try:
                        from TTS.tts.configs.xtts_config import XttsConfig
                        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
                        from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
                        torch.serialization.add_safe_globals([
                            XttsConfig, XttsAudioConfig, XttsArgs,
                            BaseDatasetConfig, BaseAudioConfig,
                        ])
                    except Exception:
                        pass  # older PyTorch versions don't have add_safe_globals

                    logger.info(
                        "[XTTS] Loading model '%s' on device '%s' "
                        "(first call triggers ~1.8 GB download if not cached)…",
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

    # ── Text splitting ────────────────────────────────────────────────────────

    # Per-language character limits copied from the XTTS v2 tokenizer.
    # We target ~60 % of each limit so segments comfortably fit.
    _CHAR_LIMITS: dict[str, int] = {
        "en": 250, "de": 253, "fr": 273, "es": 239, "it": 213,
        "pt": 203, "pl": 224, "zh-cn": 82, "ar": 166, "cs": 186,
        "ru": 182, "nl": 251, "tr": 226, "ja": 71, "hu": 224, "ko": 95,
        "hi": 250,  # not in tokenizer dict → falls back to 250
    }

    @classmethod
    def _split_text(cls, text: str, lang_code: str) -> list[str]:
        """Split *text* into model-safe segments for the given XTTS language.

        For languages where pysbd doesn't recognise sentence boundaries
        (Hindi `।`, Japanese `。！？`) we apply a regex-based pre-split so
        that no segment exceeds the model's per-language character limit.

        Returns a list with at least one element (the original text if it is
        already short enough and no special handling is needed).
        """
        import re

        limit = cls._CHAR_LIMITS.get(lang_code, 250)
        target = int(limit * 0.85)  # stay well below the hard limit

        # ── Hindi: split on Devanagari danda (।) and dandaanda (॥) ──────────
        # Use a tighter target (~40 % of limit) because Hindi preprocessing
        # in XTTS v2 is minimal (no phonemizer) — shorter segments produce
        # cleaner, less garbled output.
        if lang_code == "hi":
            hi_target = int(limit * 0.40)  # ~100 chars per segment for Hindi
            raw = re.split(r"(?<=[।॥?!])\s*", text)
            segments: list[str] = []
            current = ""
            for chunk in raw:
                chunk = chunk.strip()
                if not chunk:
                    continue
                if len(current) + len(chunk) + 1 <= hi_target:
                    current = f"{current} {chunk}".strip() if current else chunk
                else:
                    if current:
                        segments.append(current)
                    current = chunk
            if current:
                segments.append(current)
            return segments or [text]

        # ── Japanese: split on 。！？and close-brackets ────────────────────
        if lang_code == "ja":
            raw = re.split(r"(?<=[。！？」』])", text)
            segments = []
            current = ""
            for chunk in raw:
                if not chunk:
                    continue
                if len(current) + len(chunk) <= target:
                    current += chunk
                else:
                    if current:
                        segments.append(current)
                    current = chunk
            if current:
                segments.append(current)
            return segments or [text]

        # ── All other languages: return as-is, rely on built-in pysbd ───────
        return [text]

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

        # Pre-split text into short segments for languages whose sentence
        # boundaries aren't recognised by the built-in pysbd splitter.
        # Segments are synthesised individually and their waveforms concatenated.
        segments = self._split_text(text, lang_code)

        # XTTS v2 model.tts() is not thread-safe — serialise inference calls.
        # Concurrent requests (e.g. "Run All" firing EN + HI + JA at once) would
        # otherwise corrupt internal tensor buffers, producing a tensor size
        # mismatch error (a=N, b=64) in the attention layer.
        logger.info(
            "[XTTS] waiting for inference slot | voice=%s | segments=%d",
            voice, len(segments),
        )
        with self._inference_lock:
            logger.info("[XTTS] inference start | voice=%s | lang=%s", voice, lang_code)
            all_wav: list[float] = []
            for seg in segments:
                seg_wav: list[float] = model.tts(
                    text=seg,
                    speaker_wav=str(ref_path),
                    language=lang_code,
                )
                all_wav.extend(seg_wav)
            wav = all_wav

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
