# Speak Like Anyone for Social Media

> **A production-grade, fully functional AI voice app — freely given to media creators
> for voice-overs on personal and non-commercial social media projects.**

A locally deployed, broadcast-quality text-to-speech system with zero-shot voice cloning
in 17 languages. All processing runs on your own machine — no cloud subscription,
no API key, no audio data leaving your device.

---

## ⚠️ Important Disclaimer — Licensing & Permitted Use

**This app is intended for personal and non-commercial social media content creation only.**

| Component | License | Permitted use |
|---|---|---|
| **Kokoro-82M** | Apache 2.0 | Free for all uses including commercial — no restrictions |
| **XTTS v2 (Coqui TTS)** | Coqui Public Model License v1.0 | Personal and non-commercial use; commercial use only for organisations with annual revenue **≤ USD $1 million** |

### Voice Cloning — Ethical & Legal Boundaries

Voice cloning is a powerful capability and must be used responsibly:

- ✅ Clone **your own voice** for content you create
- ✅ Clone another person's voice **only with their explicit, informed consent**
- ✅ Use for **personal projects**, social media channels, podcast intros, and hobby narration
- ❌ Do **not** impersonate, deceive, or misrepresent another person
- ❌ Do **not** use for commercial gain beyond the CPML v1.0 revenue threshold
- ❌ Do **not** generate audio for fraud, manipulation, or non-consensual media

> By using this application you accept full responsibility for complying with applicable
> laws (including voice and biometric data laws in your jurisdiction) and for obtaining
> any necessary consent before cloning another person's voice.

---

## What Makes This App Unique

Most open-source TTS tools are either cloud-dependent, legally restricted for voice
cloning, or require a machine-learning research background to operate.
**Speak Like Anyone for Social Media** is different:

| Feature | Why it matters for creators |
|---|---|
| **54 built-in voices across 9 languages** | Pick the perfect narrator — British, American, Hindi, French, Spanish, Italian, Portuguese, Japanese, Chinese |
| **Zero-shot voice cloning** | Reproduce any voice from a 5–12 second sample clip — no training, no fine-tuning |
| **17-language cloning** | Narrate in English, Hindi, Japanese, French, German, Spanish, and more in the cloned voice |
| **Broadcast-grade audio** | 44.1 kHz stereo WAV · EBU R128 loudness normalisation · noise reduction · EQ — ready for DaVinci Resolve or Premiere |
| **SSML narration control** | Script precise pauses, emphasis, and character-spelling inline in your text |
| **Fully local** | No API keys, no subscriptions, no audio uploaded anywhere |
| **Runs on consumer hardware** | macOS Apple Silicon (MPS), Windows / Linux (CUDA or CPU) |

---

## Capabilities at a Glance

| Capability | Detail |
|---|---|
| **Built-in voices** | 54 voices — EN-US, EN-GB, Hindi, Spanish, French, Italian, Portuguese, Japanese, Chinese |
| **Zero-shot voice cloning** | Clone any speaker from a short reference clip (XTTS v2, 17 languages) |
| **SSML narration control** | `<pause>`, `<break>`, `<emphasis>`, `<say-as>` tags inline in script text |
| **Batch synthesis** | Submit a full script (up to 500 segments) in a single API call |
| **44.1 kHz broadcast export** | Stereo WAV at CD quality — drop directly into NLEs |
| **EBU R128 loudness profiles** | broadcast −23 LUFS · streaming −14 LUFS · podcast −16 LUFS |
| **Noise reduction** | Spectral subtraction denoising on every output |
| **Parametric EQ** | 80 Hz high-pass + 3 kHz presence shelf for narration clarity |
| **GPU acceleration** | Auto-detects CUDA (Linux/WSL2), MPS (Apple Silicon), or CPU |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   BROWSER UI (Vanilla JS)                        │
│  Text input · Voice selector · Built-in / Clone / Test Results  │
└──────────────────────┬───────────────────────────────────────────┘
                       │  FastAPI (Python)
        ┌──────────────┴──────────────┐
        │         API Layer           │
        │  SSML Preprocessor          │
        │  Engine Router              │
        └──────┬──────────────┬───────┘
               │              │
   ┌───────────▼──┐    ┌──────▼──────────────┐
   │  Kokoro-82M  │    │      XTTS v2         │
   │ (54 built-in │    │ (zero-shot cloning   │
   │   voices)    │    │  17 languages)       │
   └───────────┬──┘    └──────┬───────────────┘
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

**Kokoro-82M** (Apache 2.0) provides 54 production-quality built-in voices across nine
languages. Its 82M-parameter footprint runs comfortably on consumer hardware without
enterprise GPU requirements.

**XTTS v2** (Coqui Public Model License v1.0) handles zero-shot voice cloning in 17
languages including Hindi and Japanese, from a 5–12 second reference clip with no
fine-tuning step. Used here exclusively for personal and non-commercial social media
content creation.

---

## Project Structure

```
speak-like-anyone-for-social-media/
├── docs/
│   └── TTS_Agent_Solution_Design.docx   # Full solution design document
└── tts-agent/                           # Application source
    ├── backend/
    │   ├── app.py                       # FastAPI factory
    │   ├── api/                         # Routes + Pydantic schemas
    │   ├── engines/                     # Kokoro + XTTS v2 engine wrappers
    │   └── utils/                       # Audio, broadcast, SSML, batch, device
    ├── frontend/                        # Browser UI (HTML/CSS/JS — 3-tab layout)
    ├── configs/                         # Pydantic Settings
    ├── tests/                           # pytest suites
    ├── docker/                          # Dockerfile + Compose (Linux/WSL2 GPU)
    ├── scripts/                         # setup.sh / setup.bat, download_models.py
    ├── input_samples/                   # Pre-packaged test reference clips
    ├── Makefile
    └── README.md                        # Detailed setup & API reference
```

---

## Quick Start

### First-time setup (run once)

```bash
cd tts-agent

bash scripts/setup.sh          # creates Python 3.11 venv, installs deps, copies .env
source venv/bin/activate

# Add your HuggingFace token to .env (optional — removes download rate limits)
# HF_TOKEN=hf_your_token_here  — get a free read-only token at:
# https://huggingface.co/settings/tokens

python scripts/download_models.py   # Kokoro (~500 MB) + XTTS v2 (~1.8 GB)

make run
```

Open `http://localhost:8000` in your browser.

### Every subsequent session

```bash
source venv/bin/activate
make run
make stop    # to stop the server
```

> **Python version:** XTTS v2 requires Python 3.9–3.11. `setup.sh` automatically
> installs Python 3.11 via the deadsnakes PPA on Linux/WSL2 when needed.

For full setup instructions, Docker usage, API reference, SSML narration control, and
testing, see [tts-agent/README.md](tts-agent/README.md).

---

## Part of

**Applied AI for Societal Impact** — a collection of practical AI/ML projects targeting
real-world applications in media, conservation, and decision support.

[github.com/rommelsharma/Applied-AI-for-Societal-Impact](https://github.com/rommelsharma/Applied-AI-for-Societal-Impact)
