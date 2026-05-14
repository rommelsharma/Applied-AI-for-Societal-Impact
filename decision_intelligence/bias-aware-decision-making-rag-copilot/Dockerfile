# syntax=docker/dockerfile:1
# ── Bias-Aware Decision Copilot — Streamlit UI ────────────────────────────────
#
# Build:  docker build -t bias-copilot .
# Run:    docker run -p 8501:8501 --env-file .env bias-copilot
#
# The public corpus artefacts (data/corpora/public/**) are baked into the image.
# The private corpus (data/corpora/private/) is excluded via .dockerignore.
# Pass all secrets at runtime via --env-file or individual -e flags.

FROM python:3.12-slim

# Non-root user for Cloud Run / App Runner compatibility
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# ── system dependencies ───────────────────────────────────────────────────────
# PyMuPDF and faiss-cpu ship self-contained wheels; no extra apt packages needed.
# curl is only for the optional HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit>=1.35.0

# ── application code ──────────────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

# Install the project package in non-editable mode (no .venv inside the image)
RUN pip install --no-cache-dir .

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --silent --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "ui/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.fileWatcherType=none"]
