from typing import Literal
from pydantic import BaseModel, Field


OutputFormat = Literal["wav", "wav44", "wav44_24bit"]
LoudnessProfile = Literal["broadcast", "streaming", "podcast", "film"]


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(..., description="Voice ID from /voices, or 'clone:<name>'")
    language: str = Field("en-us", description="BCP-47 language tag")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    use_ssml: bool = Field(
        False,
        description=(
            "When true, text is parsed as SSML markup. "
            "Supported tags: <pause ms='N'/>, <break strength='paragraph|sentence'/>, "
            "<emphasis level='strong|moderate|reduced'>…</emphasis>, "
            "<say-as interpret-as='characters'>…</say-as>"
        ),
    )
    reference_audio: str | None = Field(None, description="Filename in voice_samples/ for F5-TTS")
    reference_text: str | None = Field(None, description="Transcript of the reference clip")
    # v2.0 additions
    output_format: OutputFormat = Field(
        "wav",
        description="wav=24kHz mono (default) | wav44=44.1kHz stereo 16-bit | wav44_24bit=44.1kHz stereo 24-bit",
    )
    loudness_profile: LoudnessProfile = Field(
        "broadcast",
        description="EBU R128 target: broadcast=-23 LUFS | streaming=-14 | podcast=-16 | film=-24",
    )
    noise_reduce: bool = Field(True, description="Apply spectral noise reduction")
    eq: bool = Field(True, description="Apply high-pass + presence EQ")


class BatchSegment(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str
    language: str = "en-us"
    speed: float = Field(1.0, ge=0.5, le=2.0)
    reference_audio: str | None = None
    reference_text: str | None = None


class BatchSynthesizeRequest(BaseModel):
    segments: list[BatchSegment] = Field(..., min_length=1, max_length=500)
    output_format: OutputFormat = "wav44"
    loudness_profile: LoudnessProfile = "broadcast"
    noise_reduce: bool = True
    eq: bool = True
    concatenate: bool = Field(False, description="Merge all segments into a single output WAV")
    gap_ms: int = Field(300, ge=0, le=5000, description="Silence between segments when concatenating")


class VoiceInfo(BaseModel):
    id: str
    name: str
    language: str
    engine: str
    reference_file: str | None = None


class SynthesizeResponse(BaseModel):
    audio_url: str
    filename: str
    duration_seconds: float
    engine: str
    output_format: str
    sample_rate: int


class BatchJobResponse(BaseModel):
    job_id: str
    status: str
    total: int
    completed: int
    failed: int
    results: list[dict]
    concat_url: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    kokoro_ready: bool
    f5_ready: bool
    device: str
    version: str = "2.2.0"
