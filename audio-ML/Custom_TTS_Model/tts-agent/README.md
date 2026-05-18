# TTS Agent

Local text-to-speech application using **Kokoro-82M** (built-in English & Hindi voices) and **F5-TTS** (zero-shot voice cloning). Both are commercially usable under Apache 2.0 / MIT licenses.

## Stack

| Component | Technology |
|---|---|
| Built-in TTS | Kokoro-82M (Apache 2.0) |
| Voice cloning | F5-TTS (MIT) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Inference | PyTorch (CUDA · MPS · CPU) |
| Packaging | Docker + Docker Compose |

## Quick Start

### Local setup (Mac or Windows WSL2)

```bash
cd tts-agent
bash scripts/setup.sh          # creates venv, installs deps, copies .env
source venv/bin/activate

python scripts/download_models.py   # pre-downloads Kokoro + F5-TTS weights

uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000
```

Or with Make:
```bash
make setup && source venv/bin/activate
make download-models
make run
```

### Docker (GPU — Linux / WSL2 with NVIDIA Container Toolkit)

```bash
# Build
make docker-build

# Run with GPU + mounted volumes
make docker-run

# Or with Compose (recommended for repeatable operation)
make docker-compose
```

> **macOS note**: Docker on Mac does not support GPU passthrough. Run natively with `make run` to use MPS on Apple Silicon.

## Device auto-detection

The `DEVICE=auto` setting in `.env` picks:
- `cuda` — NVIDIA GPU on Linux / Windows WSL2
- `mps`  — Apple Silicon GPU (M1/M2/M3/M4)
- `cpu`  — fallback (slower, no GPU needed)

Override by setting `DEVICE=cpu` in `.env`.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Browser UI |
| GET | `/health` | Engine status + device |
| GET | `/voices` | All available voices |
| POST | `/synthesize` | Generate speech (returns JSON with audio URL) |
| GET | `/audio/{filename}` | Serve generated WAV |
| POST | `/upload-voice-sample` | Upload reference clip for cloning |
| GET | `/voice-samples` | List uploaded reference clips |
| DELETE | `/voice-samples/{filename}` | Remove a reference clip |
| GET | `/docs` | FastAPI interactive docs |

### Synthesize — request body

```json
{
  "text": "Hello, this is a test.",
  "voice": "af_bella",
  "language": "en-us",
  "speed": 1.0
}
```

For voice cloning (F5-TTS), set `voice` to `clone:<name>` and provide `reference_audio`:

```json
{
  "text": "Text to synthesize in the cloned voice.",
  "voice": "clone:myspeaker",
  "language": "en-us",
  "speed": 1.0,
  "reference_audio": "myspeaker.wav",
  "reference_text": "Transcript of the reference clip."
}
```

### cURL examples

```bash
# List voices
curl http://localhost:8000/voices

# Synthesize with Kokoro built-in voice
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a test.","voice":"af_bella","language":"en-us"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['audio_url'])"

# Download audio
curl http://localhost:8000/audio/<filename>.wav --output output.wav

# Upload voice sample for cloning
curl -X POST http://localhost:8000/upload-voice-sample \
  -F "file=@my_voice.wav"
```

## Testing

```bash
make test
make test-cov
```

## Built-in voices

| Voice ID | Description |
|---|---|
| `af_bella` | American English Female |
| `af_nicole` | American English Female |
| `af_sarah` | American English Female |
| `af_sky` | American English Female |
| `am_adam` | American English Male |
| `am_michael` | American English Male |
| `bf_emma` | British English Female |
| `bf_isabella` | British English Female |
| `bm_george` | British English Male |
| `bm_lewis` | British English Male |
| `hf_alpha` | Hindi Female |
| `hf_beta` | Hindi Female |
| `hm_omega` | Hindi Male |

## Porting to another machine

1. Install NVIDIA drivers + Docker with GPU support (Linux/WSL2)
2. `docker load < tts-agent.tar.gz` or clone the repo
3. Copy `.env` and `models/` directory (or re-run `make download-models`)
4. `make docker-compose`

## Project structure

```
tts-agent/
├── backend/
│   ├── app.py              # FastAPI app factory
│   ├── api/
│   │   ├── routes.py       # All route handlers
│   │   └── schemas.py      # Pydantic request/response models
│   ├── engines/
│   │   ├── base.py         # Abstract engine contract
│   │   ├── kokoro_engine.py
│   │   └── f5_engine.py
│   └── utils/
│       ├── audio.py        # Resample, normalize, trim
│       ├── device.py       # CUDA/MPS/CPU detection
│       └── logger.py
├── frontend/
│   ├── templates/index.html
│   └── static/{css,js}/
├── configs/settings.py     # Pydantic settings from .env
├── tests/
├── docker/
├── scripts/
├── Makefile
├── requirements.txt
└── .env.example
```
