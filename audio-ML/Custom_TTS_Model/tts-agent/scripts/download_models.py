"""
Download model weights for Kokoro-82M and F5-TTS.

Kokoro-82M — weights are fetched automatically by the kokoro package
from HuggingFace Hub on first use, but this script pre-downloads them
so the container or offline machine doesn't need internet access later.

F5-TTS — similarly auto-downloads on first use via the f5-tts package.
"""
import sys
from pathlib import Path

# Ensure project root is in path when run as `python scripts/download_models.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.utils.logger import get_logger
from configs.settings import settings

logger = get_logger("download_models")


def download_kokoro():
    logger.info("Pre-downloading Kokoro-82M weights…")
    try:
        from kokoro import KPipeline
        # Instantiating triggers weight download from HuggingFace Hub
        for lang_code in ("a", "b", "h"):
            logger.info("  Loading lang_code='%s'", lang_code)
            KPipeline(lang_code=lang_code)
        logger.info("Kokoro weights ready.")
    except ImportError:
        logger.error("kokoro package not installed. Run: pip install kokoro>=0.9.4")
    except Exception as exc:
        logger.error("Kokoro download failed: %s", exc)


def download_f5():
    logger.info("Pre-downloading F5-TTS weights…")
    try:
        from f5_tts.api import F5TTS
        F5TTS(model_type=settings.F5_MODEL_NAME)
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
