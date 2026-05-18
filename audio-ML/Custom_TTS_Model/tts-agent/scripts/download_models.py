"""
Download model weights for Kokoro-82M and F5-TTS.

Kokoro-82M — weights are fetched automatically by the kokoro package
from HuggingFace Hub on first use, but this script pre-downloads them
so the container or offline machine doesn't need internet access later.

F5-TTS — similarly auto-downloads on first use via the f5-tts package.

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
        # huggingface_hub also reads HUGGING_FACE_HUB_TOKEN as a fallback
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("HuggingFace token configured — authenticated downloads enabled.")
        except ImportError:
            pass  # huggingface_hub not yet installed; env var is still set
    else:
        logger.warning(
            "HF_TOKEN not set. Downloads may be rate-limited. "
            "Add HF_TOKEN=hf_... to your .env file."
        )


def download_kokoro():
    _configure_hf_token()
    # _configure_espeak() must run before any kokoro/phonemizer import so that
    # ESPEAK_DATA_PATH is set before the espeak C library initialises.
    from backend.engines.kokoro_engine import _configure_espeak, _KOKORO_REPO_ID
    _configure_espeak()
    logger.info("Pre-downloading Kokoro-82M weights…")
    try:
        from kokoro import KPipeline
        # Instantiating triggers weight download from HuggingFace Hub
        for lang_code in ("a", "b", "h"):
            logger.info("  Loading lang_code='%s'", lang_code)
            KPipeline(lang_code=lang_code, repo_id=_KOKORO_REPO_ID)
        logger.info("Kokoro weights ready.")
    except ImportError:
        logger.error("kokoro package not installed. Run: pip install kokoro>=0.9.4")
    except Exception as exc:
        logger.error("Kokoro download failed: %s", exc)


def download_f5():
    _configure_hf_token()
    logger.info("Pre-downloading F5-TTS weights…")
    try:
        from f5_tts.api import F5TTS
        F5TTS(model=settings.F5_MODEL_NAME)
        logger.info("F5-TTS weights ready.")
    except ImportError:
        logger.error("f5-tts package not installed. Run: pip install f5-tts")
    except Exception as exc:
        logger.error("F5-TTS download failed: %s", exc)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-download TTS model weights")
    parser.add_argument("--kokoro", action="store_true", help="Download Kokoro only")
    parser.add_argument("--f5", action="store_true", help="Download F5-TTS only")
    args = parser.parse_args()

    if args.kokoro:
        download_kokoro()
    elif args.f5:
        download_f5()
    else:
        download_kokoro()
        download_f5()
