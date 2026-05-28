#!/usr/bin/env bash
# One-command local setup for macOS (Apple Silicon or Intel) and Windows WSL2
set -euo pipefail

# Coqui TTS (XTTS v2) requires Python <3.12. Prefer 3.11 when available.

# Detect WSL early (needed by _pick_python before PLATFORM is set)
_is_wsl() {
  grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null || \
  [ -f /proc/sys/fs/binfmt_misc/WSLInterop ]
}

# On WSL/Linux, install Python 3.11 via deadsnakes PPA if only 3.12+ is present
_ensure_python311_wsl() {
  if ! _is_wsl; then return; fi
  if command -v python3.11 &>/dev/null; then return; fi  # already installed

  local default_ver
  default_ver=$(python3 -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
  if [ "$default_ver" != "(3, 12)" ] && [ "$default_ver" != "(3, 13)" ]; then return; fi

  echo "WSL detected: default Python is ${default_ver} (incompatible with XTTS v2)."
  echo "Installing Python 3.11 via deadsnakes PPA…"

  if ! command -v apt-get &>/dev/null; then
    echo "WARNING: apt-get not found — cannot auto-install Python 3.11. Install it manually." >&2
    return
  fi

  sudo apt-get update -qq
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -qq
  sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils

  echo "Python 3.11 installed successfully."
}

_pick_python() {
  # On WSL, guarantee 3.11 is present before we search
  _ensure_python311_wsl

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

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment…"
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
  # Linux / WSL2: prefer CUDA 12.1 build; falls back to CPU if driver absent
  if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected — installing CUDA 12.1 PyTorch…"
    pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cu121
  else
    echo "No GPU detected — installing CPU PyTorch…"
    pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "Installing application dependencies…"
pip install --quiet -r requirements.txt

echo "Downloading UniDic dictionary for Japanese TTS support…"
python -m unidic download --quiet 2>/dev/null || python -m unidic download || true

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