"""
Pre-download model weights for Kokoro-82M and XTTS v2.

Kokoro-82M — weights are fetched automatically by the kokoro package
from HuggingFace Hub on first use, but this script pre-downloads them
so the container or offline machine doesn't need internet access later.

XTTS v2 — similarly auto-downloads on first use via Coqui TTS (~1.8 GB).
Pre-downloading is strongly recommended before first server start to avoid
a long pause on the first synthesis request.

HuggingFace token
-----------------
Set HF_TOKEN in your .env file (copied from .env.example) to remove
rate limits and get faster downloads. A read-only token is sufficient.
Get one at: https://huggingface.co/settings/tokens
"""
import os
import sys
from pathlib import Path

# Ensure project root is in path when run as `python scripts/download_models.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils.logger import get_logger
from configs.settings import settings

logger = get_logger("download_models")


def _configure_hf_token() -> None:
    """Export HF_TOKEN to the environment so huggingface_hub picks it up automatically."""
    token = settings.HF_TOKEN
    if token and token != "hf_your_token_here":
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("HuggingFace token configured — authenticated downloads enabled.")
        except ImportError:
            pass
    else:
        logger.warning(
            "HF_TOKEN not set. Downloads may be rate-limited. "
            "Add HF_TOKEN=hf_... to your .env file."
        )


def download_kokoro() -> None:
    _configure_hf_token()
    from backend.engines.kokoro_engine import _configure_espeak, _KOKORO_REPO_ID
    from backend.utils.device import DEVICE   # respects CUDA kernel-check + CPU fallback
    _configure_espeak()
    logger.info("Pre-downloading Kokoro-82M weights… (device=%s)", DEVICE)
    try:
        from kokoro import KPipeline
        # Core lang_codes always downloaded:
        #   a=EN-US  b=EN-GB  h=Hindi
        #   e=Spanish  f=French  i=Italian  p=Portuguese
        for lang_code in ("a", "b", "h", "e", "f", "i", "p"):
            logger.info("  Loading lang_code='%s'", lang_code)
            KPipeline(lang_code=lang_code, repo_id=_KOKORO_REPO_ID, device=DEVICE)
        # CJK lang_codes require optional misaki extras
        for lang_code, pkg in (("j", "misaki[ja]"), ("z", "misaki[zh]")):
            try:
                logger.info("  Loading lang_code='%s'", lang_code)
                KPipeline(lang_code=lang_code, repo_id=_KOKORO_REPO_ID, device=DEVICE)
            except Exception as e:
                logger.warning(
                    "  lang_code='%s' skipped — install %s to enable: %s",
                    lang_code, pkg, e,
                )
        logger.info("Kokoro weights ready.")
    except ImportError:
        logger.error("kokoro package not installed. Run: pip install kokoro>=0.9.4")
    except Exception as exc:
        logger.error("Kokoro download failed: %s", exc)


def download_xtts() -> None:
    """Trigger XTTS v2 model download via Coqui TTS (≈ 1.8 GB).

    The TTS library downloads the model to:
      macOS/Linux:  ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/
      Windows:      %LOCALAPPDATA%\\tts\\tts_models--multilingual--multi-dataset--xtts_v2\\
    """
    _configure_hf_token()
    logger.info("Pre-downloading XTTS v2 weights (~1.8 GB)…")
    try:
        import torch
        from TTS.api import TTS
        from backend.utils.device import DEVICE

        # PyTorch 2.6+ changed torch.load default to weights_only=True, which
        # blocks Coqui TTS checkpoint classes. Register them as trusted globals
        # before instantiating TTS so the download + verification load succeeds.
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            from TTS.config.shared_configs import BaseDatasetConfig, BaseAudioConfig
            torch.serialization.add_safe_globals([
                XttsConfig, XttsAudioConfig, XttsArgs,
                BaseDatasetConfig, BaseAudioConfig,
            ])
        except Exception:
            pass  # older PyTorch — add_safe_globals not needed

        logger.info("  Downloading to Coqui TTS cache (device=%s)…", DEVICE)
        # Instantiating TTS triggers the download; no synthesis needed
        tts = TTS(settings.XTTS_MODEL_NAME)
        logger.info("XTTS v2 weights ready.")
        # Free memory — model is not needed beyond download verification
        del tts
    except ImportError:
        logger.error(
            "Coqui TTS package not installed. Run: pip install TTS>=0.22.0"
        )
    except Exception as exc:
        logger.error("XTTS v2 download failed: %s", exc)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-download TTS model weights")
    parser.add_argument("--kokoro", action="store_true", help="Download Kokoro only")
    parser.add_argument("--xtts",   action="store_true", help="Download XTTS v2 only")
    args = parser.parse_args()

    if args.kokoro:
        download_kokoro()
    elif args.xtts:
        download_xtts()
    else:
        download_kokoro()
        download_xtts()
