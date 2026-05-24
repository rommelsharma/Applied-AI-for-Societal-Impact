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

    # Pre-packaged reference audio for the built-in test suite.
    # Place the following files here before clicking "Run All Tests":
    #   cloning-voice-clip-male-hindi-1.wav   (Hindi male reference)
    #   cloning-voice-samples-JP.wav          (Japanese reference)
    # Files are served via GET /input-samples/{filename} and are accepted
    # as reference_audio values by the XTTS engine (checked before voice_samples/).
    INPUT_SAMPLES_DIR: Path = Path("input_samples")

    # Kokoro engine
    KOKORO_LANG_EN: str = "a"   # American English
    KOKORO_LANG_HI: str = "h"   # Hindi

    # XTTS v2 engine — multilingual zero-shot voice cloning
    # Supported languages: en es fr de it pt pl tr ru nl cs ar zh-cn hu ko ja hi
    # The model (~1.8 GB) auto-downloads on first synthesis call via Coqui TTS.
    # Pre-download with:  python scripts/download_models.py --xtts
    # License: Coqui Public Model License v1.0 (commercial use ≤ $1M/yr revenue)
    XTTS_MODEL_NAME: str = "tts_models/multilingual/multi-dataset/xtts_v2"

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
    def input_samples_dir(self) -> Path:
        p = self.abs_path(self.INPUT_SAMPLES_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def kokoro_model_dir(self) -> Path:
        return self.abs_path(self.MODEL_DIR / "kokoro")

    @property
    def xtts_model_dir(self) -> Path:
        return self.abs_path(self.MODEL_DIR / "xtts")


settings = Settings()
