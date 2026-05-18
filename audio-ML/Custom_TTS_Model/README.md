# Custom TTS Model — Applied AI for Societal Impact

A production-grade, locally deployed text-to-speech system built for documentary narration,
training material production, and professional media workflows. The system prioritises
**commercial licensing safety**, **broadcast-quality audio**, and **local privacy** — all
inference runs on your machine with no telemetry or cloud dependency.

---

## Purpose

Narration-quality TTS is typically either expensive (commercial cloud APIs) or
legally restricted (many open-source models prohibit commercial output). This project
provides an end-to-end solution that is:

- **Commercially usable** — all models are Apache 2.0 or MIT licensed
- **Broadcast-ready** — audio post-processing meets EBU R128 / ITU-R BS.1770 loudness standards
- **Locally private** — voice samples and generated audio never leave the host machine
- **Cross-platform** — runs on Windows WSL2 (CUDA) and macOS (MPS / CPU) without reconfiguration

---

## Capabilities

| Capability | Detail |
|---|---|
| **Built-in English voices** | 10 voices (American + British, male/female) via Kokoro-82M |
| **Built-in Hindi voices** | 3 voices via Kokoro-82M |
| **Zero-shot voice cloning** | Clone any speaker from a short reference clip via F5-TTS |
| **SSML narration control** | Pause, emphasis, and character-spelling tags inline in script text |
| **Batch synthesis** | Submit a full documentary script (up to 500 segments) in a single API call |
| **44.1 kHz broadcast export** | Stereo WAV at CD quality — accepted directly by NLEs (DaVinci Resolve, Premiere) |
| **EBU R128 loudness normalisation** | Target profiles: broadcast −23 LUFS, streaming −14 LUFS, podcast −16 LUFS |
| **Noise reduction** | Spectral subtraction denoising on every output |
| **Parametric EQ** | 80 Hz high-pass + 3 kHz presence shelf for narration clarity |
| **GPU acceleration** | Auto-detects CUDA (Linux/WSL2), MPS (Apple Silicon), or CPU |
| **Docker packaging** | Reproducible deployment with host-mounted model and output volumes |

---

## Solution Approach

```
┌────────────────────────────────────────────────────────────────────┐
│                     BROWSER UI (Vanilla JS)                        │
│  Text input · Voice selector · Built-in / Clone mode · Batch      │
└────────────────────────┬───────────────────────────────────────────┘
                         │  FastAPI (Python)
          ┌──────────────┴──────────────┐
          │       API Layer             │
          │  SSML Preprocessor          │
          │  Engine Router              │
          └──────┬──────────────┬───────┘
                 │              │
     ┌───────────▼──┐    ┌──────▼──────────┐
     │  Kokoro-82M  │    │    F5-TTS        │
     │  (built-in   │    │  (voice cloning  │
     │   voices)    │    │   from clip)     │
     └───────────┬──┘    └──────┬───────────┘
                 └──────┬───────┘
                        ▼
          ┌─────────────────────────┐
          │  Audio Post-Processing  │
          │  · Resample → 44.1 kHz  │
          │  · Noise reduction      │
          │  · EQ (HP + presence)   │
          │  · EBU R128 loudness    │
          │  · True-peak limiter    │
          │  · Mono → Stereo        │
          └─────────────┬───────────┘
                        ▼
                  outputs/*.wav
```

### Model selection rationale

**Kokoro-82M** (Apache 2.0) is chosen for its strong built-in voice quality, efficient
local inference, and broad language support (English + Hindi). Its small footprint (82M
parameters) makes it practical on laptops without enterprise GPU hardware.

**F5-TTS** (MIT) is chosen for the cloning path as a commercially safe replacement for
XTTS-v2 (whose license restricts revenue-generating usage). Flow-matching architecture
gives it strong zero-shot cloning from short reference clips.

### SSML markup

Text submitted to the API may include lightweight markup tags for narration control:

```xml
Good morning.<pause ms="500"/>

Today we explore <emphasis level="strong">climate change</emphasis>
and its effects.<break strength="paragraph"/>

The agency <say-as interpret-as="characters">IPCC</say-as> reports…
```

Tags are parsed before synthesis; silence blocks are inserted in the waveform
post-synthesis; emphasis is realised by adjusting the synthesis speed of each tagged span.

### Batch synthesis

Submit a full documentary script in one call:

```json
POST /batch-synthesize
{
  "segments": [
    { "text": "Chapter one.", "voice": "af_bella" },
    { "text": "The journey begins.", "voice": "af_bella" }
  ],
  "output_format": "wav44",
  "loudness_profile": "broadcast",
  "concatenate": true,
  "gap_ms": 500
}
```

Returns a `job_id`; poll `/batch-status/{job_id}` for the manifest and the
optional concatenated output URL.

---

## Project Structure

```
Custom_TTS_Model/
├── docs/
│   └── TTS_Agent_Solution_Design.docx   # Full solution design document
└── tts-agent/                           # Application source
    ├── backend/
    │   ├── app.py                       # FastAPI factory
    │   ├── api/                         # Routes + Pydantic schemas
    │   ├── engines/                     # Kokoro + F5-TTS wrappers
    │   └── utils/                       # Audio, broadcast, SSML, batch, device
    ├── frontend/                        # Browser UI (HTML/CSS/JS)
    ├── configs/                         # Pydantic Settings
    ├── tests/                           # pytest suites
    ├── docker/                          # Dockerfile + Compose
    ├── scripts/                         # setup.sh, download_models.py
    ├── Makefile
    └── README.md                        # Detailed setup & API reference
```

---

## Quick Start

### First-time setup (run once)

```bash
cd tts-agent

bash scripts/setup.sh          # creates venv, installs deps, copies .env
source venv/bin/activate

# Add your HuggingFace token to .env (removes download rate limits)
# HF_TOKEN=hf_your_token_here  — get one at https://huggingface.co/settings/tokens

python scripts/download_models.py   # downloads Kokoro (~500 MB) + F5-TTS (~1.2 GB)
# or individually:  --kokoro  /  --f5

make run
```

Open `http://localhost:8000` in your browser.

### Every subsequent session

```bash
source venv/bin/activate
make run
```

> The message `HF_TOKEN is set and is the current active token` during model download is **normal** — it confirms authentication is working.

For full setup, Docker usage, API reference, SSML narration control, and testing, see
[tts-agent/README.md](tts-agent/README.md).

---

## Licensing

| Component | License | Commercial use |
|---|---|---|
| Kokoro-82M | Apache 2.0 | Yes |
| F5-TTS | MIT | Yes |
| pyloudnorm | MIT | Yes |
| noisereduce | MIT | Yes |
| FastAPI / Uvicorn | MIT | Yes |
| PyTorch | BSD-3 | Yes |

All components permit commercial use including monetised documentary production
and broadcast distribution.

---

## Part of

**Applied AI for Societal Impact** — a collection of practical AI/ML projects
targeting real-world applications in media, conservation, and decision support.
