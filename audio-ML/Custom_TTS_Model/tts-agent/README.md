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

### Step 1 — First-time setup (run once)

```bash
cd tts-agent

# Creates venv, installs all dependencies, copies .env.example → .env
bash scripts/setup.sh          # or: make setup

# Activate the virtual environment
source venv/bin/activate
```

Then open `.env` and add your HuggingFace token (removes rate limits):
```
HF_TOKEN=hf_your_token_here
```
Get a free read-only token at https://huggingface.co/settings/tokens

### Step 2 — Download model weights (run once, ~1.7 GB total)

```bash
# Download both engines
python scripts/download_models.py   # or: make download-models

# Download individual engines if preferred
python scripts/download_models.py --kokoro   # Kokoro-82M only (~500 MB)
python scripts/download_models.py --f5       # F5-TTS only (~1.2 GB)
```

> **Note:** You will see the message `HF_TOKEN is set and is the current active token independently from the token you've just configured.` — this is normal and confirms authentication is working.

### Step 3 — Start the server

```bash
make run
# Open http://localhost:8000
```

---

### Every subsequent session (two commands)

```bash
source venv/bin/activate
make run
```

---

### Docker (GPU — Linux / WSL2 with NVIDIA Container Toolkit)

```bash
make docker-build
make docker-run       # or: make docker-compose
```

> **macOS note**: Docker on Mac does not support GPU passthrough. Run natively with `make run` to use MPS on Apple Silicon.

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
  "speed": 1.0,
  "use_ssml": false,
  "output_format": "wav",
  "loudness_profile": "broadcast",
  "noise_reduce": true,
  "eq": true
}
```

| Field | Default | Description |
|---|---|---|
| `text` | required | Input text (plain or SSML-tagged) |
| `voice` | required | Voice ID from `/voices`, or `clone:<name>` for cloning |
| `language` | `"en-us"` | BCP-47 language tag |
| `speed` | `1.0` | Playback speed (0.5–2.0) |
| `use_ssml` | `false` | Parse text as SSML markup (see SSML section below) |
| `output_format` | `"wav"` | `wav` = 24 kHz mono · `wav44` = 44.1 kHz stereo 16-bit · `wav44_24bit` = 44.1 kHz stereo 24-bit |
| `loudness_profile` | `"broadcast"` | EBU R128 target: `broadcast` (−23 LUFS) · `streaming` (−14) · `podcast` (−16) · `film` (−24) |
| `noise_reduce` | `true` | Spectral noise reduction |
| `eq` | `true` | 80 Hz high-pass + 3 kHz presence shelf |

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

## SSML Narration Control

Set `use_ssml: true` in the API request (or select **SSML Markup** in the UI dropdown) to enable markup-based narration control.

| Tag | Effect |
|---|---|
| `<pause ms="500"/>` | Insert 500 ms of silence (50–5000 ms) |
| `<break strength="paragraph"/>` | 800 ms silence (`sentence` = 350 ms, `medium` = 200 ms) |
| `<emphasis level="strong">…</emphasis>` | Synthesise span at 0.85× speed (slower = more emphatic) |
| `<emphasis level="moderate">…</emphasis>` | 0.92× speed |
| `<emphasis level="reduced">…</emphasis>` | 1.10× speed |
| `<say-as interpret-as="characters">NASA</say-as>` | Spell out letter by letter |

**Sample file:** `tests/sample_ssml.txt` — paste this into the UI to test a full documentary narration.

```xml
Welcome to the documentary narration.<pause ms="600"/>

Today we explore <emphasis level="strong">climate change</emphasis>
and its consequences.<break strength="paragraph"/>

The <say-as interpret-as="characters">IPCC</say-as> reports temperatures
have risen by <emphasis level="moderate">1.1 degrees Celsius</emphasis>
since pre-industrial times.<pause ms="400"/>
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
│   ├── app.py                  # FastAPI app factory
│   ├── api/
│   │   ├── routes.py           # All route handlers
│   │   └── schemas.py          # Pydantic request/response models
│   ├── engines/
│   │   ├── base.py             # Abstract engine contract
│   │   ├── kokoro_engine.py    # Kokoro-82M wrapper + espeak fix
│   │   └── f5_engine.py        # F5-TTS zero-shot cloning wrapper
│   └── utils/
│       ├── audio.py            # Resample, normalise, trim
│       ├── broadcast.py        # EBU R128, noise reduction, EQ, 44.1 kHz export
│       ├── ssml.py             # SSML parser (pause, break, emphasis, say-as)
│       ├── batch.py            # Async batch synthesis job runner
│       ├── device.py           # CUDA / MPS / CPU detection
│       └── logger.py
├── frontend/
│   ├── templates/index.html    # Browser UI (plain text & SSML modes)
│   └── static/{css,js}/
├── configs/settings.py         # Pydantic settings loaded from .env
├── tests/
│   ├── sample_ssml.txt         # SSML narration demo — paste into UI to test
│   └── test_*.py
├── docker/
├── scripts/
│   ├── setup.sh                # One-command env setup
│   └── download_models.py      # Pre-download Kokoro + F5-TTS weights
├── Makefile
├── requirements.txt
└── .env.example
```
