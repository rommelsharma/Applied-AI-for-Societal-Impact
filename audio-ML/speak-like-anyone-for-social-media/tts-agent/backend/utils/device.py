import logging
import torch
from configs.settings import settings

logger = logging.getLogger(__name__)


def _cuda_kernels_work() -> bool:
    """Return True if CUDA kernels actually execute on this GPU.

    torch.cuda.is_available() only confirms a driver is present — it does NOT
    verify that the installed PyTorch wheels contain kernels compiled for the
    current GPU architecture.  RTX 50xx (Blackwell, sm_12x) cards raise
    "no kernel image is available for execution on the device" at runtime when
    paired with pre-cu126 PyTorch builds.  Running a tiny op here lets us catch
    that early and fall back to CPU instead of crashing mid-synthesis.
    """
    try:
        t = torch.zeros(1, device="cuda")
        _ = t + t          # forces a CUDA kernel dispatch
        return True
    except Exception:
        return False


def get_device() -> str:
    """Return the best available inference device respecting .env DEVICE setting."""
    device = settings.DEVICE.lower()
    if device != "auto":
        return device

    if torch.cuda.is_available():
        if _cuda_kernels_work():
            return "cuda"
        # GPU is visible but kernels won't run — likely an architecture mismatch
        # (e.g. RTX 50xx Blackwell with a pre-cu126 PyTorch build).
        cc = ""
        try:
            major, minor = torch.cuda.get_device_capability()
            cc = f" (compute capability {major}.{minor})"
        except Exception:
            pass
        gpu_name = ""
        try:
            gpu_name = f" [{torch.cuda.get_device_name(0)}]"
        except Exception:
            pass
        logger.warning(
            "CUDA is available%s%s but kernel execution failed — "
            "the installed PyTorch build likely lacks kernels for this GPU architecture. "
            "Re-run setup.bat (Windows) or setup.sh (Linux/WSL2) to install the "
            "correct PyTorch for your GPU (RTX 50xx needs cu126+). "
            "Falling back to CPU.",
            gpu_name, cc,
        )
        return "cpu"

    # Apple Silicon GPU via Metal Performance Shaders
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


DEVICE = get_device()
