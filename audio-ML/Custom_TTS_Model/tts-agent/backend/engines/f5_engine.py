"""F5-TTS engine wrapper for zero-shot voice cloning (MIT-licensed — commercial-safe)."""
from pathlib import Path

import numpy as np

from backend.engines.base import TTSEngine
from backend.utils.audio import postprocess, audio_to_bytes, load_audio
from backend.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)


class F5Engine(TTSEngine):
    """Wraps F5-TTS for zero-shot voice cloning from user-supplied reference audio."""

    def __init__(self) -> None:
        self._model = None
        self._ready = False
        self._load()

    def _load(self) -> None:
        try:
            from f5_tts.api import F5TTS
            self._F5TTS = F5TTS
            self._ready = True
            logger.info("F5-TTS engine initialized (model loads on first synthesis call)")
        except ImportError as exc:
            logger.error("f5-tts package not installed: %s", exc)

    def _get_model(self):
        if self._model is None:
            logger.info("Loading F5-TTS model weights…")
            self._model = self._F5TTS(model=settings.F5_MODEL_NAME)
        return self._model

    def synthesize(
        self,
        text: str,
        voice: str,  # unused for F5 — uses reference_audio instead
        language: str = "en-us",
        speed: float = 1.0,
        reference_audio: str | None = None,
        reference_text: str | None = None,
        **kwargs,
    ) -> bytes:
        if not self._ready:
            raise RuntimeError("F5-TTS engine failed to initialize. Check installation.")

        if not reference_audio:
            raise ValueError("F5-TTS requires a reference_audio path for voice cloning.")

        ref_path = settings.voice_samples_dir / reference_audio
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")

        model = self._get_model()
        logger.info("F5-TTS synthesis | ref=%s | chars=%d", reference_audio, len(text))

        wav, sr, _ = model.infer(
            ref_file=str(ref_path),
            ref_text=reference_text or "",
            gen_text=text,
            speed=speed,
        )

        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        processed, out_sr = postprocess(audio, src_rate=sr)
        return audio_to_bytes(processed, out_sr)

    def list_voices(self) -> list[dict]:
        """Return voice entries built from voice_samples/ directory."""
        samples_dir = settings.voice_samples_dir
        voices = []
        for p in sorted(samples_dir.iterdir()):
            if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
                voices.append(
                    {
                        "id": f"clone:{p.stem}",
                        "name": f"{p.stem} (Cloned)",
                        "language": "any",
                        "engine": "f5-tts",
                        "reference_file": p.name,
                    }
                )
        return voices

    def is_ready(self) -> bool:
        return self._ready
