from abc import ABC, abstractmethod


class TTSEngine(ABC):
    """Contract that every synthesis engine must satisfy."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str,
        language: str = "en-us",
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Return raw WAV bytes for the synthesized audio."""

    @abstractmethod
    def list_voices(self) -> list[dict]:
        """Return list of voice descriptors: {id, name, language, engine}."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the engine has loaded its model successfully."""
