import torch
from configs.settings import settings


def get_device() -> str:
    """Return the best available inference device respecting .env DEVICE setting."""
    device = settings.DEVICE.lower()
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"

    # Apple Silicon GPU via Metal Performance Shaders
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


DEVICE = get_device()
