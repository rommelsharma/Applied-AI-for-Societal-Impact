#!/usr/bin/env bash
# One-command local setup for macOS (Apple Silicon or Intel) and Windows WSL2
set -euo pipefail

PYTHON=${PYTHON:-python3}
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

echo "Installing dev dependencies…"
pip install --quiet -r requirements-dev.txt

# Create .env if missing
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit as needed."
fi

# Create runtime dirs
mkdir -p models/kokoro models/f5tts voice_samples outputs

echo ""
echo "=== Setup complete ==="
echo "Activate env:  source venv/bin/activate"
echo "Download models: python scripts/download_models.py"
echo "Start server:    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"
echo "Open browser:    http://localhost:8000"
