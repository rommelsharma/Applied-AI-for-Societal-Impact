#!/usr/bin/env bash
# One-command local setup for macOS (Apple Silicon or Intel) and Linux / WSL2
set -euo pipefail

# Coqui TTS (XTTS v2) requires Python <3.12. Prefer 3.11 when available.

# On Linux (including WSL2), install Python 3.11 via deadsnakes PPA if the
# system default is 3.12+.  Works regardless of whether WSL is detectable.
_ensure_python311_linux() {
  if [ "$(uname -s)" != "Linux" ]; then return; fi
  if command -v python3.11 &>/dev/null; then return; fi  # already installed

  local major minor
  major=$(python3 -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo 0)
  minor=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo 0)

  # Only intervene when default Python is 3.12+
  if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 12 ]; }; then
    return
  fi

  echo "Default Python is ${major}.${minor} — incompatible with XTTS v2 (requires <3.12)."
  echo "Installing Python 3.11 via deadsnakes PPA…"

  if ! command -v apt-get &>/dev/null; then
    echo "ERROR: apt-get not found. Install Python 3.11 manually, then re-run this script." >&2
    exit 1
  fi

  sudo apt-get update -qq
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -qq
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

  echo "Python 3.11 installed."
}

_pick_python() {
  # On Linux, guarantee 3.11 is present before searching
  _ensure_python311_linux

  for candidate in python3.11 \
      /opt/anaconda3/envs/PyTorchEnv/bin/python3.11 \
      /opt/anaconda3/envs/PyTorchEnv/bin/python \
      python3 python; do
    if command -v "$candidate" &>/dev/null; then
      ver=$("$candidate" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
      if [ "$ver" = "(3, 11)" ] || [ "$ver" = "(3, 10)" ] || [ "$ver" = "(3, 9)" ]; then
        echo "$candidate"; return
      fi
    fi
  done
  # Fallback — warn user
  echo "python3"
  echo "WARNING: Could not find Python 3.9–3.11. TTS (XTTS v2) requires Python <3.12." >&2
}
PYTHON=${PYTHON:-$(_pick_python)}
VENV_DIR="venv"

echo "=== TTS Agent Setup ==="
echo "Python interpreter: $PYTHON ($($PYTHON --version 2>&1))"

# Detect platform
OS="$(uname -s)"
case "$OS" in
  Darwin)   PLATFORM="macos" ;;
  Linux)    PLATFORM="linux" ;;
  *)        echo "Unsupported OS: $OS"; exit 1 ;;
esac

echo "Platform: $PLATFORM"

# Install system espeak-ng so phonemizer finds it at a short, fixed path.
# The bundled espeakng_loader path can exceed the espeak C library's 160-char buffer limit,
# causing a double-slash "//phontab" error at runtime.
if [ "$PLATFORM" = "macos" ]; then
  if ! [ -d "/opt/homebrew/share/espeak-ng-data" ] && \
     ! [ -d "/opt/homebrew/lib/espeak-ng-data" ]  && \
     ! [ -d "/usr/local/share/espeak-ng-data" ]; then
    echo "Installing espeak-ng via Homebrew (required by Kokoro / phonemizer)…"
    if command -v brew &>/dev/null; then
      brew install espeak-ng
    else
      echo "WARNING: Homebrew not found. Install espeak-ng manually: brew install espeak-ng"
    fi
  else
    echo "espeak-ng already installed."
  fi
elif [ "$PLATFORM" = "linux" ]; then
  if ! [ -d "/usr/share/espeak-ng-data" ] && ! [ -d "/usr/lib/espeak-ng-data" ]; then
    echo "Installing espeak-ng via apt (required by Kokoro / phonemizer)…"
    if command -v apt-get &>/dev/null; then
      sudo apt-get install -y espeak-ng ffmpeg
    elif command -v dnf &>/dev/null; then
      sudo dnf install -y espeak-ng
    else
      echo "WARNING: Could not detect apt or dnf. Install espeak-ng manually."
    fi
  else
    echo "espeak-ng already installed."
  fi
fi

# Create virtual environment — recreate if it exists but uses the wrong Python version
if [ -d "$VENV_DIR" ]; then
  existing_ver=$("$VENV_DIR/bin/python" -c "import sys; print(sys.version_info[:2])" 2>/dev/null || echo "unknown")
  expected_ver=$($PYTHON -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
  if [ "$existing_ver" != "$expected_ver" ]; then
    echo "Existing venv uses Python $existing_ver but need $expected_ver — recreating…"
    rm -rf "$VENV_DIR"
  else
    echo "Existing venv is correct Python $existing_ver — reusing."
  fi
fi
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment with $PYTHON…"
  $PYTHON -m venv "$VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip…"
pip install --quiet --upgrade pip wheel

# Install PyTorch with the right backend
if [ "$PLATFORM" = "macos" ]; then
  # macOS: standard PyPI build supports MPS on Apple Silicon
  echo "Installing PyTorch (macOS — MPS supported on Apple Silicon)…"
  pip install --quiet torch torchaudio
else
  # Linux / WSL2: detect GPU architecture; RTX 50xx (Blackwell, sm_12x) needs cu126
  if command -v nvidia-smi &>/dev/null; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    CC_MAJOR=$(echo "${CC:-0}" | cut -d. -f1)
    if [ "${CC_MAJOR:-0}" -ge 12 ] 2>/dev/null; then
      echo "NVIDIA Blackwell GPU detected (compute capability ${CC}) — installing CUDA 12.6 PyTorch…"
      echo "NOTE: RTX 50xx requires CUDA 12.6+ wheels. Installing cu126 build."
      pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cu126
    else
      echo "NVIDIA GPU detected (compute capability ${CC}) — installing CUDA 12.1 PyTorch…"
      pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
  else
    echo "No GPU detected — installing CPU PyTorch…"
    pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "Installing application dependencies…"
pip install --quiet -r requirements.txt

# Kokoro CJK language support — optional extras that require a C++ compiler.
# misaki[ja] → pyopenjtalk (Japanese phonemisation)
# misaki[zh] → jieba + pypinyin (Chinese/Mandarin phonemisation)
# Both are skipped gracefully if the build fails (e.g. cmake not found).
echo "Installing Kokoro Japanese support (misaki[ja])…"
if pip install --quiet 'misaki[ja]' 2>/dev/null; then
  echo "  misaki[ja] installed."
else
  echo "  WARNING: misaki[ja] install failed — Japanese Kokoro voices will be unavailable."
  echo "           macOS fix:  brew install cmake && pip install 'misaki[ja]'"
  echo "           Linux fix:  sudo apt-get install -y cmake build-essential && pip install 'misaki[ja]'"
fi

echo "Installing Kokoro Chinese support (misaki[zh])…"
if pip install --quiet 'misaki[zh]' 2>/dev/null; then
  echo "  misaki[zh] installed."
else
  echo "  WARNING: misaki[zh] install failed — Chinese Kokoro voices will be unavailable."
fi

if python -c "import unidic, os; assert os.path.isdir(unidic.DICDIR)" 2>/dev/null; then
  echo "UniDic already installed — skipping download."
else
  echo "Downloading UniDic dictionary for Japanese TTS support…"
  python -m unidic download --quiet 2>/dev/null || python -m unidic download || true
fi

echo "Installing dev dependencies…"
pip install --quiet -r requirements-dev.txt

# Create .env if missing
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit as needed."
fi

# Create runtime dirs
mkdir -p models/kokoro models/xtts voice_samples input_samples outputs

echo ""
echo "=== Setup complete ==="
echo "Activate env:       source venv/bin/activate"
echo "Download models:    python scripts/download_models.py"
echo "Start server:       uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"
echo "Open browser:       http://localhost:8000"
echo ""
echo "Test suite files (already in input_samples/ if repo was cloned):"
echo "  cloning-voice-sample-english-male-1.wav"
echo "  cloning-voice-sample-hindi-male-1.wav"
echo "  cloning-voice-sample-japanese-female-1.wav"