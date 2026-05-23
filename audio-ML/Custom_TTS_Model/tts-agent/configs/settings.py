from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # HuggingFace token for authenticated downloads (removes rate limits)
    HF_TOKEN: str | None = None

    # Inference device: auto | cuda | mps | cpu
    DEVICE: str = "auto"

    # Directories (relative to project root)
    BASE_DIR: Path = Path(__file__).resolve().parents[1]
    MODEL_DIR: Path = Path("models")
    OUTPUT_DIR: Path = Path("outputs")
    VOICE_SAMPLES_DIR: Path = Path("voice_samples")

    # Kokoro engine
    KOKORO_LANG_EN: str = "a"   # American English
    KOKORO_LANG_HI: str = "h"   # Hindi

    # F5-TTS engine — base model (English + Mandarin)
    F5_MODEL_NAME: str = "F5TTS_v1_Base"
    F5_VOCODER_NAME: str = "vocos"

    # ── Language-specific F5-TTS checkpoints ─────────────────────────────────
    # Models trained on specific scripts that understand their native character
    # sets directly (no IPA phonemisation needed).
    # Set any *_MODEL_NAME to "" to disable that override and fall back to the
    # base model + espeak-ng phonemisation for that language.
    #
    # Hindi — SPRINGLab/F5-Hindi-24KHz (Devanagari vocab, 73 chars)
    F5_HINDI_MODEL_NAME: str = "SPRINGLab/F5-Hindi-24KHz"
    F5_HINDI_CKPT_FILE: str = ""          # empty = auto-detect (first .safetensors/.pt)
    F5_HINDI_VOCAB_FILE: str = ""         # empty = auto-detect (vocab.txt in repo root)

    # Japanese — Jmica/F5TTS, JA_21999120 checkpoint
    # (Hiragana + Katakana + Kanji vocab, 2976 entries)
    # Architecture: F5TTS_Small (depth=18) — Jmica was fine-tuned from the
    # original Small, not v1_Base, so F5TTS_Small is the correct arch config.
    F5_JAPANESE_MODEL_NAME: str = "Jmica/F5TTS"
    F5_JAPANESE_CKPT_FILE: str = "JA_21999120/model_21999120.pt"
    F5_JAPANESE_VOCAB_FILE: str = "JA_21999120/vocab_japanese.txt"
    F5_JAPANESE_ARCH_MODEL: str = "F5TTS_Small"  # must match checkpoint depth

    # Audio post-processing
    TARGET_SAMPLE_RATE: int = 24000
    TARGET_LOUDNESS_DB: float = -20.0

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    def abs_path(self, rel: Path) -> Path:
        return (self.BASE_DIR / rel).resolve()

    @property
    def output_dir(self) -> Path:
        p = self.abs_path(self.OUTPUT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def voice_samples_dir(self) -> Path:
        p = self.abs_path(self.VOICE_SAMPLES_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def kokoro_model_dir(self) -> Path:
        return self.abs_path(self.MODEL_DIR / "kokoro")

    @property
    def f5_model_dir(self) -> Path:
        return self.abs_path(self.MODEL_DIR / "f5tts")


settings = Settings()
