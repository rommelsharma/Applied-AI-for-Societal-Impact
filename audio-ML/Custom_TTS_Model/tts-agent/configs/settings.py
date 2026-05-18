from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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

    # F5-TTS engine
    F5_MODEL_NAME: str = "F5-TTS"
    F5_VOCODER_NAME: str = "vocos"

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
