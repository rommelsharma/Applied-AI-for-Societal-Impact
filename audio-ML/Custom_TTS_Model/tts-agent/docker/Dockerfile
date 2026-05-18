# TTS Agent — GPU-enabled Docker image
# Requires NVIDIA Container Toolkit for GPU access
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY backend/    ./backend/
COPY frontend/   ./frontend/
COPY configs/    ./configs/
COPY scripts/    ./scripts/

# Runtime directories (override with host mounts)
RUN mkdir -p models/kokoro models/f5tts voice_samples outputs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
