# TTS Agent

Local text-to-speech application using **Kokoro-82M** (built-in English & Hindi voices) and **XTTS v2** (zero-shot voice cloning in 17 languages). Commercially usable under Apache 2.0 and Coqui Public Model License v1.0.

## Stack

| Component | Technology |
|---|---|
| Built-in TTS | Kokoro-82M (Apache 2.0) |
| Voice cloning | XTTS v2 — Coqui TTS (CPML v1.0, commercial ≤ $1M/yr) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (3-tab: Built-in / Clone / Test Results) |
| Inference | PyTorch (CUDA · MPS · CPU) |
| Packaging | Docker + Docker Compose |

## Quick Start

### Step 1 — First-time setup (run once)

```bash
cd tts-agent

# Creates Python 3.11 venv, installs all dependencies, copies .env.example → .env
# On macOS this also installs espeak-ng via Homebrew (required by Kokoro)
bash scripts/setup.sh          # or: make setup

# Activate the virtual environment
source venv/bin/activate
```

> **Python version:** XTTS v2 requires Python 3.9–3.11. `setup.sh` auto-selects a compatible
> Python interpreter. If running manually, use `python3.11` — do not use `python3` if it
> resolves to 3.12 or later on your system.

> **WSL2 users:** Run `setup.sh` from the Windows filesystem path (`/mnt/c/Users/<you>/...`) where GitHub Desktop manages the repo, not from a separate WSL clone. This keeps git changes in sync automatically.

> **macOS users:** `setup.sh` automatically runs `brew install espeak-ng`.
> Without this, Kokoro's phonemizer backend cannot find its data files.

> **Japanese support:** `setup.sh` automatically downloads the UniDic dictionary
> (`python -m unidic download`, ~500 MB) required for Japanese TTS with XTTS v2.
> Without it, Japanese synthesis fails with a MeCab initialization error.

Then open `.env` and add your HuggingFace token (removes rate limits):
```
HF_TOKEN=hf_your_token_here
```
Get a free read-only token at https://huggingface.co/settings/tokens

### Step 2 — Download model weights (run once, ~2.3 GB total)

```bash
# Download both engines
python scripts/download_models.py   # or: make download-models

# Download individual engines if preferred
python scripts/download_models.py --kokoro   # Kokoro-82M only (~500 MB)
python scripts/download_models.py --xtts    # XTTS v2 only (~1.8 GB)

# Download Japanese dictionary (required for Japanese voice cloning)
python -m unidic download
```

> XTTS v2 downloads to `~/.local/share/tts/` (macOS/Linux) or `%LOCALAPPDATA%\tts\` (Windows).
> Mount this as a Docker volume to persist across container restarts.

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

### Stop the server

```bash
Ctrl+C          # if running in the foreground
make stop       # if running in the background
```

---

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
| GET | `/input-samples` | List pre-packaged test audio clips |
| GET | `/input-samples/{filename}` | Serve a pre-packaged test clip |
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
| `language` | `"en-us"` | BCP-47 language tag (XTTS v2 supports: en, hi, ja, zh-cn, fr, de, es, it, pt, pl, tr, ru, nl, cs, ar, hu, ko) |
| `speed` | `1.0` | Playback speed (0.5–2.0) |
| `use_ssml` | `false` | Parse text as SSML markup (see SSML section below) |
| `output_format` | `"wav"` | `wav` = 24 kHz mono · `wav44` = 44.1 kHz stereo 16-bit · `wav44_24bit` = 44.1 kHz stereo 24-bit |
| `loudness_profile` | `"broadcast"` | EBU R128 target: `broadcast` (−23 LUFS) · `streaming` (−14) · `podcast` (−16) · `film` (−24) |
| `noise_reduce` | `true` | Spectral noise reduction |
| `eq` | `true` | 80 Hz high-pass + 3 kHz presence shelf |

For voice cloning (XTTS v2), set `voice` to `clone:<name>` and provide `reference_audio`:

```json
{
  "text": "Text to synthesize in the cloned voice.",
  "voice": "clone:myspeaker",
  "language": "hi",
  "speed": 1.0,
  "reference_audio": "myspeaker.wav"
}
```

The engine searches for the reference file in `voice_samples/` first, then `input_samples/`.
No `reference_text` is needed — XTTS v2 is fully zero-shot.

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

## Test Results tab

The UI includes a dedicated **Test Results** tab with three pre-configured voice cloning tests:

| # | Test | Reference clip | Language | Expected output |
|---|---|---|---|---|
| 1 | English voice cloning | `cloning-voice-sample-english-male-1.wav` | English (en) | Fluent English narration in cloned voice |
| 2 | Hindi voice cloning | `cloning-voice-sample-hindi-male-1.wav` | Hindi (hi) | Fluent Hindi narration in cloned voice |
| 3 | Japanese voice cloning | `cloning-voice-sample-japanese-female-1.wav` | Japanese (ja) | Fluent Japanese narration in cloned voice |

All three files are version-controlled in `input_samples/` — no manual placement needed after cloning the repo.
See `input_samples/README.md` for recording guidelines and reference transcripts.

## Testing

```bash
make test
make test-cov
```

## Built-in voices

| Voice ID | Description |
|---|---|
| `af_heart` | American English Female |
| `af_bella` | American English Female |
| `af_aoede` | American English Female |
| `af_kore` | American English Female |
| `af_alloy` | American English Female |
| `af_nova` | American English Female |
| `af_nicole` | American English Female |
| `af_sarah` | American English Female |
| `af_sky` | American English Female |
| `am_fenrir` | American English Male |
| `am_michael` | American English Male |
| `am_puck` | American English Male |
| `am_echo` | American English Male |
| `am_eric` | American English Male |
| `am_liam` | American English Male |
| `am_onyx` | American English Male |
| `am_adam` | American English Male |
| `bf_emma` | British English Female |
| `bf_isabella` | British English Female |
| `bf_alice` | British English Female |
| `bf_lily` | British English Female |
| `bm_george` | British English Male |
| `bm_fable` | British English Male |
| `bm_daniel` | British English Male |
| `bm_lewis` | British English Male |
| `hf_alpha` | Hindi Female |
| `hf_beta` | Hindi Female |
| `hm_omega` | Hindi Male |
| `hm_psi` | Hindi Male |

## Porting to another machine

1. Install NVIDIA drivers + Docker with GPU support (Linux/WSL2)
2. `docker load < tts-agent.tar.gz` or clone the repo
3. Copy `.env` and mount `~/.local/share/tts/` volume (or re-run `make download-models`)
4. `make docker-compose`

## Project structure

```
tts-agent/
├── backend/
│   ├── app.py                  # FastAPI app factory (v3.0.0)
│   ├── api/
│   │   ├── routes.py           # All route handlers
│   │   └── schemas.py          # Pydantic request/response models
│   ├── engines/
│   │   ├── base.py             # Abstract engine contract
│   │   ├── kokoro_engine.py    # Kokoro-82M wrapper + espeak fix
│   │   └── xtts_engine.py      # XTTS v2 zero-shot cloning wrapper
│   └── utils/
│       ├── audio.py            # Resample, normalise, trim
│       ├── broadcast.py        # EBU R128, noise reduction, EQ, 44.1 kHz export
│       ├── ssml.py             # SSML parser (pause, break, emphasis, say-as)
│       ├── batch.py            # Async batch synthesis job runner
│       ├── device.py           # CUDA / MPS / CPU detection
│       └── logger.py
├── frontend/
│   ├── templates/index.html    # 3-tab Browser UI
│   └── static/{css,js}/
├── configs/settings.py         # Pydantic settings loaded from .env
├── input_samples/              # Pre-packaged Hindi + Japanese test reference clips
│   └── README.md               # Recording guidelines + test specifications
├── tests/
│   ├── sample_ssml.txt         # SSML narration demo — paste into UI to test
│   └── test_*.py
├── docker/
├── scripts/
│   ├── setup.sh                # One-command env setup (macOS / Linux / WSL2)
│   ├── setup.bat               # One-command env setup (Windows native)
│   └── download_models.py      # Pre-download Kokoro + XTTS v2 weights
├── Makefile
├── requirements.txt
└── .env.example
```
