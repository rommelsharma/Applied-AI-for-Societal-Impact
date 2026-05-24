import asyncio
import io
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse

from backend.api.schemas import (
    SynthesizeRequest,
    SynthesizeResponse,
    BatchSynthesizeRequest,
    BatchJobResponse,
    VoiceInfo,
    HealthResponse,
)
from backend.engines.kokoro_engine import KokoroEngine
from backend.engines.xtts_engine import XTTSEngine
from backend.utils.audio import postprocess
from backend.utils.broadcast import broadcast_postprocess
from backend.utils.ssml import parser as ssml_parser, make_silence, TextSegment, SilenceSegment
from backend.utils.batch import create_job, get_job, run_batch, MAX_SEGMENTS
from backend.utils.device import DEVICE
from backend.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)
router = APIRouter()

# Engine singletons — initialised once at import time
_kokoro = KokoroEngine()
_xtts = XTTSEngine()

# Thread pool for CPU-bound batch work — keeps the event loop unblocked
_batch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch")


def _engine_synthesize(req_dict: dict) -> tuple[bytes, str]:
    """Shared synthesis helper used by both single and batch paths."""
    voice = req_dict.get("voice", "")
    is_clone = voice.startswith("clone:")

    if is_clone:
        if not _xtts.is_ready():
            raise RuntimeError("XTTS v2 engine is not available. Run: pip install TTS>=0.22.0")
        wav_bytes = _xtts.synthesize(
            text=req_dict["text"],
            voice=voice,
            language=req_dict.get("language", "en-us"),
            speed=req_dict.get("speed", 1.0),
            reference_audio=req_dict.get("reference_audio"),
            reference_text=req_dict.get("reference_text"),
        )
        return wav_bytes, "xtts-v2"
    else:
        if not _kokoro.is_ready():
            raise RuntimeError("Kokoro engine is not available.")
        wav_bytes = _kokoro.synthesize(
            text=req_dict["text"],
            voice=voice,
            language=req_dict.get("language", "en-us"),
            speed=req_dict.get("speed", 1.0),
        )
        return wav_bytes, "kokoro"


def _apply_ssml_and_synthesize(req: SynthesizeRequest) -> tuple[np.ndarray, int, str]:
    """
    Parse SSML markup (when use_ssml=True), synthesise each TextSegment,
    insert silence for SilenceSegments, and return the assembled waveform.
    When use_ssml=False the text is sent directly to the engine.
    """
    if not req.use_ssml:
        wav_bytes, engine = _engine_synthesize(req.model_dump())
        buf = io.BytesIO(wav_bytes)
        audio, sr = sf.read(buf, dtype="float32")
        return audio, sr, engine

    segments = ssml_parser.parse(req.text)

    # If no recognised SSML tags were found even in SSML mode, synthesise plain
    has_ssml = any(isinstance(s, SilenceSegment) or
                   (isinstance(s, TextSegment) and s.speed_override != 1.0)
                   for s in segments)

    if not has_ssml:
        wav_bytes, engine = _engine_synthesize(req.model_dump())
        buf = io.BytesIO(wav_bytes)
        audio, sr = sf.read(buf, dtype="float32")
        return audio, sr, engine

    # SSML path: synthesise each text segment independently
    chunks: list[np.ndarray] = []
    sr_out = settings.TARGET_SAMPLE_RATE

    # Set the engine default correctly based on the voice type so the label
    # is accurate even if no text segments are synthesised (all-silence edge case).
    engine = "xtts-v2" if req.voice.startswith("clone:") else "kokoro"

    for seg in segments:
        if isinstance(seg, SilenceSegment):
            chunks.append(make_silence(seg.duration_ms, sr_out))
        elif isinstance(seg, TextSegment):
            # Skip fragments that are too short to produce meaningful audio.
            # Very short segments (lone punctuation, stray whitespace after tag
            # extraction) cause garbled output or silence from TTS engines.
            stripped = seg.text.strip()
            if len(stripped) < 2:
                logger.debug("SSML: skipping too-short segment %r", stripped)
                continue

            req_dict = req.model_dump()
            req_dict["text"] = stripped
            req_dict["speed"] = req.speed * seg.speed_override
            req_dict["speed"] = max(0.5, min(req_dict["speed"], 2.0))
            try:
                wav_bytes, engine = _engine_synthesize(req_dict)
            except Exception as exc:
                logger.warning("SSML segment synthesis failed for %r: %s", stripped[:40], exc)
                # Skip the failed segment but continue with the rest
                continue
            buf = io.BytesIO(wav_bytes)
            seg_audio, seg_sr = sf.read(buf, dtype="float32")
            # Ensure consistent sample rate across all segments
            if seg_sr != sr_out:
                from backend.utils.audio import resample
                seg_audio = resample(seg_audio, seg_sr, sr_out)
            chunks.append(seg_audio)

    combined = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return combined, sr_out, engine


def _apply_output_format(
    audio: np.ndarray,
    sr: int,
    req: SynthesizeRequest,
) -> tuple[np.ndarray, int, str]:
    """Apply broadcast pipeline or basic postprocess depending on output_format."""
    if req.output_format in ("wav44", "wav44_24bit"):
        audio, sr = broadcast_postprocess(
            audio, sr,
            loudness_profile=req.loudness_profile,
            noise_reduce=req.noise_reduce,
            eq=req.eq,
        )
        subtype = "PCM_24" if req.output_format == "wav44_24bit" else "PCM_16"
    else:
        audio, sr = postprocess(audio, sr)
        subtype = "PCM_16"
    return audio, sr, subtype


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        kokoro_ready=_kokoro.is_ready(),
        xtts_ready=_xtts.is_ready(),
        device=DEVICE,
    )


@router.get("/voices", response_model=list[VoiceInfo])
def voices():
    all_voices = _kokoro.list_voices() + _xtts.list_voices()
    return [VoiceInfo(**v) for v in all_voices]


# ── Input samples (pre-packaged test references) ──────────────────────────────

@router.get("/input-samples")
def list_input_samples():
    """List audio files in the input_samples/ directory (pre-packaged test references)."""
    import soundfile as _sf
    result = []
    for p in sorted(settings.input_samples_dir.iterdir()):
        if p.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
            continue
        entry: dict = {"filename": p.name, "size_bytes": p.stat().st_size}
        try:
            info = _sf.info(str(p))
            entry["duration_seconds"] = round(info.duration, 2)
        except Exception:
            entry["duration_seconds"] = None
        result.append(entry)
    return result


@router.get("/input-samples/{filename}")
def serve_input_sample(filename: str):
    """Serve a file from input_samples/ (used by the Test Results audio players)."""
    safe_name = Path(filename).name   # strip any ../ traversal
    path = settings.input_samples_dir / safe_name
    if not path.exists() or path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
        raise HTTPException(404, "Input sample not found.")
    return FileResponse(str(path), media_type="audio/wav")


@router.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    start = time.perf_counter()

    # ── Request traceability log ──────────────────────────────────────────
    engine_hint = "xtts-v2" if req.voice.startswith("clone:") else "kokoro"
    logger.info(
        "[SYNTH] request | engine=%s | voice=%s | lang=%s | "
        "chars=%d | format=%s | ssml=%s | speed=%.1f | loudness=%s",
        engine_hint, req.voice, req.language,
        len(req.text), req.output_format, req.use_ssml,
        req.speed, req.loudness_profile,
    )

    try:
        audio, sr, engine = _apply_ssml_and_synthesize(req)
        audio, sr, subtype = _apply_output_format(audio, sr, req)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("[SYNTH] failed | engine=%s | lang=%s | error=%s", engine_hint, req.language, exc)
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        logger.error("[SYNTH] failed | engine=%s | lang=%s | error=%s", engine_hint, req.language, exc)
        raise HTTPException(500, str(exc))
    except Exception as exc:
        logger.error(
            "[SYNTH] unexpected error | engine=%s | lang=%s | error=%s",
            engine_hint, req.language, exc, exc_info=True,
        )
        raise HTTPException(500, f"Synthesis error: {exc}")

    filename = f"{uuid.uuid4().hex}.wav"
    out_path = settings.output_dir / filename
    sf.write(str(out_path), audio, sr, subtype=subtype)

    duration = (len(audio) / sr) if audio.ndim == 1 else (audio.shape[0] / sr)
    elapsed = time.perf_counter() - start

    # ── Output confirmation log ───────────────────────────────────────────
    logger.info(
        "[SYNTH] output | engine=%s | voice=%s | lang=%s | "
        "duration=%.2fs | sample_rate=%d | format=%s | subtype=%s | elapsed=%.2fs | file=%s",
        engine, req.voice, req.language,
        duration, sr, req.output_format, subtype, elapsed, filename,
    )

    return SynthesizeResponse(
        audio_url=f"/audio/{filename}",
        filename=filename,
        duration_seconds=round(duration, 3),
        engine=engine,
        output_format=req.output_format,
        sample_rate=sr,
    )


@router.post("/batch-synthesize", response_model=BatchJobResponse)
async def batch_synthesize(req: BatchSynthesizeRequest, background_tasks: BackgroundTasks):
    if len(req.segments) > MAX_SEGMENTS:
        raise HTTPException(400, f"Maximum {MAX_SEGMENTS} segments per batch.")

    job = create_job(total_segments=len(req.segments))
    use_broadcast = req.output_format in ("wav44", "wav44_24bit")
    segment_dicts = [s.model_dump() for s in req.segments]

    loop = asyncio.get_event_loop()
    background_tasks.add_task(
        loop.run_in_executor,
        _batch_executor,
        lambda: run_batch(
            job=job,
            requests=segment_dicts,
            synthesize_fn=_engine_synthesize,
            gap_ms=req.gap_ms,
            concatenate=req.concatenate,
            loudness_profile=req.loudness_profile,
            broadcast=use_broadcast,
        ),
    )

    logger.info("Batch job %s queued: %d segments", job.job_id, len(req.segments))
    return BatchJobResponse(**job.progress)


@router.get("/batch-status/{job_id}", response_model=BatchJobResponse)
def batch_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return BatchJobResponse(**job.progress)


@router.get("/audio/{filename}")
def serve_audio(filename: str):
    safe_name = Path(filename).name  # strip any ../ path traversal components
    path = settings.output_dir / safe_name
    if not path.exists() or path.suffix != ".wav":
        raise HTTPException(404, "Audio file not found.")
    return FileResponse(str(path), media_type="audio/wav")


@router.post("/upload-voice-sample")
async def upload_voice_sample(file: UploadFile = File(...)):
    allowed = {".wav", ".mp3", ".flac", ".ogg"}
    safe_name = Path(file.filename).name  # strip any ../ path traversal components
    suffix = Path(safe_name).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(allowed)}")
    dest = settings.voice_samples_dir / safe_name
    content = await file.read()
    dest.write_bytes(content)

    # Check clip duration and warn if outside XTTS v2 optimal range (3–12 s)
    warning: str | None = None
    try:
        import io as _io
        import soundfile as _sf
        buf = _io.BytesIO(content)
        info = _sf.info(buf)
        duration_s = info.duration
        if duration_s < 3:
            warning = f"Clip is {duration_s:.1f}s — too short. XTTS v2 needs at least 3 s of clear speech."
        elif duration_s > 15:
            warning = (
                f"Clip is {duration_s:.1f}s — longer than the recommended 15 s maximum. "
                "Trim to 5–12 s of clean, single-speaker speech for best results."
            )
        logger.info("Uploaded voice sample: %s (%.1fs, %d bytes)", safe_name, duration_s, len(content))
    except Exception:
        logger.info("Uploaded voice sample: %s (%d bytes)", safe_name, len(content))

    result: dict = {"filename": safe_name, "size_bytes": len(content)}
    if warning:
        result["warning"] = warning
    return result


@router.get("/voice-samples")
def list_voice_samples():
    import soundfile as _sf
    samples_dir = settings.voice_samples_dir
    result = []
    for p in sorted(samples_dir.iterdir()):
        if p.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
            continue
        entry: dict = {"filename": p.name, "size_bytes": p.stat().st_size}
        try:
            info = _sf.info(str(p))
            entry["duration_seconds"] = round(info.duration, 2)
        except Exception:
            entry["duration_seconds"] = None
        result.append(entry)
    return result


@router.delete("/voice-samples/{filename}")
def delete_voice_sample(filename: str):
    safe_name = Path(filename).name  # strip any ../ path traversal components
    path = settings.voice_samples_dir / safe_name
    if not path.exists():
        raise HTTPException(404, "Voice sample not found.")
    path.unlink()
    return {"deleted": safe_name}
