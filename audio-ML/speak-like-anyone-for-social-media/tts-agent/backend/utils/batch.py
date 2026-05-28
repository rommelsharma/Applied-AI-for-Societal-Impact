"""
Batch synthesis engine.

Accepts a list of SynthesizeRequest objects, processes them sequentially
(to stay within GPU memory budget), and writes results to outputs/.

Supports an optional concatenated output with configurable inter-segment silence.
"""
from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from backend.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)

MAX_SEGMENTS = 500
DEFAULT_GAP_MS = 300   # silence between concatenated segments


class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETE  = "complete"
    FAILED    = "failed"


@dataclass
class SegmentResult:
    index: int
    filename: str
    audio_url: str
    duration_seconds: float
    engine: str
    char_count: int
    elapsed_seconds: float
    error: str | None = None


@dataclass
class BatchJob:
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.PENDING
    total: int = 0
    completed: int = 0
    failed: int = 0
    results: list[SegmentResult] = field(default_factory=list)
    concat_url: str | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def progress(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "results": [vars(r) for r in self.results],
            "concat_url": self.concat_url,
            "error": self.error,
        }


# In-memory job store (sufficient for local deployment)
_jobs: dict[str, BatchJob] = {}


def get_job(job_id: str) -> BatchJob | None:
    return _jobs.get(job_id)


def create_job(total_segments: int) -> BatchJob:
    job = BatchJob(total=total_segments)
    _jobs[job.job_id] = job
    return job


def run_batch(
    job: BatchJob,
    requests: list[dict],
    synthesize_fn: Callable[[dict], tuple[bytes, str]],
    *,
    gap_ms: int = DEFAULT_GAP_MS,
    concatenate: bool = False,
    loudness_profile: str = "broadcast",
    broadcast: bool = False,
) -> None:
    """
    Execute a batch job synchronously (call from a background thread/task).

    synthesize_fn(request_dict) → (wav_bytes, engine_name)
    """
    from backend.utils.audio import audio_to_bytes
    from backend.utils.broadcast import broadcast_postprocess

    job.status = JobStatus.RUNNING
    job.started_at = time.monotonic()

    all_chunks: list[np.ndarray] = []
    gap_samples_sr: tuple[int, int] | None = None  # (n_samples, sample_rate)

    for i, req in enumerate(requests):
        t0 = time.perf_counter()
        try:
            wav_bytes, engine = synthesize_fn(req)

            buf = io.BytesIO(wav_bytes)
            audio, sr = sf.read(buf, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if broadcast:
                audio, sr = broadcast_postprocess(
                    audio, sr, loudness_profile=loudness_profile
                )

            filename = f"{uuid.uuid4().hex}.wav"
            out_path = settings.output_dir / filename
            sf.write(str(out_path), audio, sr, subtype="PCM_16")

            duration = len(audio) / sr
            elapsed = time.perf_counter() - t0

            result = SegmentResult(
                index=i,
                filename=filename,
                audio_url=f"/audio/{filename}",
                duration_seconds=round(duration, 3),
                engine=engine,
                char_count=len(req.get("text", "")),
                elapsed_seconds=round(elapsed, 3),
            )
            job.results.append(result)
            job.completed += 1

            if concatenate:
                all_chunks.append(audio)
                if gap_samples_sr is None:
                    gap_samples_sr = (int(sr * gap_ms / 1000), sr)

            logger.info("Batch [%d/%d] %.2fs | %s | %s", i + 1, job.total, duration, engine, filename)

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Batch segment %d failed: %s", i, exc)
            job.results.append(
                SegmentResult(
                    index=i,
                    filename="",
                    audio_url="",
                    duration_seconds=0.0,
                    engine="",
                    char_count=len(req.get("text", "")),
                    elapsed_seconds=round(elapsed, 3),
                    error=str(exc),
                )
            )
            job.failed += 1

    # Optional concatenation
    if concatenate and all_chunks:
        gap_sr = gap_samples_sr[1] if gap_samples_sr else settings.TARGET_SAMPLE_RATE
        gap_n  = gap_samples_sr[0] if gap_samples_sr else int(gap_sr * gap_ms / 1000)
        silence = np.zeros(gap_n, dtype=np.float32)

        parts: list[np.ndarray] = []
        for idx, chunk in enumerate(all_chunks):
            # Ensure mono for concatenation (broadcast produces stereo)
            mono = chunk[:, 0] if chunk.ndim == 2 else chunk
            parts.append(mono)
            if idx < len(all_chunks) - 1:
                parts.append(silence)

        combined = np.concatenate(parts)
        concat_filename = f"batch_{job.job_id}.wav"
        concat_path = settings.output_dir / concat_filename
        sf.write(str(concat_path), combined, gap_sr, subtype="PCM_16")
        job.concat_url = f"/audio/{concat_filename}"
        logger.info("Concatenated batch → %s (%.1fs)", concat_filename, len(combined) / gap_sr)

    job.status = JobStatus.COMPLETE if job.failed == 0 else JobStatus.FAILED
    job.finished_at = time.monotonic()
    elapsed_total = job.finished_at - (job.started_at or job.finished_at)
    logger.info(
        "Batch %s complete: %d ok / %d failed in %.1fs",
        job.job_id, job.completed, job.failed, elapsed_total
    )
