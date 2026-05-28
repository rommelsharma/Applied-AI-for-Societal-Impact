# TTS Agent — GPU-enabled Docker image
# Requires NVIDIA Container Toolkit for GPU access on Linux / WSL2.
#
# XTTS v2 note: the model (~1.8 GB) auto-downloads to ~/.local/share/tts/
# on first synthesis if not pre-cached.  Mount a volume there for persistence:
#   docker run -v tts_cache:/root/.local/share/tts ...
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download UniDic dictionary for Japanese TTS support (MeCab tokenizer data, ~500 MB)
RUN python -m unidic download

# Application source
COPY backend/    ./backend/
COPY frontend/   ./frontend/
COPY configs/    ./configs/
COPY scripts/    ./scripts/

# Runtime directories (override with host mounts)
RUN mkdir -p models/kokoro models/xtts voice_samples input_samples outputs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
