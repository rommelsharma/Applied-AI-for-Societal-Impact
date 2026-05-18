"""
SSML-style markup pre-processor for narration control.

Supported tags
--------------
<pause ms="N"/>                   — insert N ms of silence (50–5000)
<break strength="paragraph"/>     — 800 ms silence
<break strength="sentence"/>      — 350 ms silence
<emphasis level="strong">…</emphasis>   — synthesise at 0.85× speed
<emphasis level="moderate">…</emphasis> — synthesise at 0.92× speed
<emphasis level="reduced">…</emphasis>  — synthesise at 1.10× speed
<say-as interpret-as="characters">…</say-as> — spell out character by character

Tags are parsed into a list of Segment objects; the synthesiser walks the list
and assembles the final waveform from audio chunks and silence blocks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Named break durations in milliseconds
BREAK_STRENGTHS: dict[str, int] = {
    "paragraph": 800,
    "sentence":  350,
    "medium":    200,
    "weak":      100,
    "none":      0,
}

EMPHASIS_SPEEDS: dict[str, float] = {
    "strong":   0.85,
    "moderate": 0.92,
    "reduced":  1.10,
    "default":  1.00,
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TextSegment:
    """A span of plain text to synthesise."""
    text: str
    speed_override: float = 1.0


@dataclass
class SilenceSegment:
    """A block of silence to insert (milliseconds)."""
    duration_ms: int


Segment = TextSegment | SilenceSegment


# ── Parser ────────────────────────────────────────────────────────────────────

class SSMLParser:
    """Parse an SSML-tagged string into a flat list of Segments."""

    # Match any tag we handle; unrecognised tags are stripped
    _TAG_RE = re.compile(
        r'<(?P<tag>[a-z\-]+)'        # opening tag name
        r'(?P<attrs>[^>/]*)'          # attributes
        r'(?:/>'                      # self-closing
        r'|>(?P<inner>[^<]*)</[a-z\-]+>)'  # wrapping content
        r'|<break\s[^/]*/?>',         # break shorthand
        re.IGNORECASE,
    )

    _ATTR_RE = re.compile(r'(\w[\w\-]*)=["\']([^"\']*)["\']')

    def _parse_attrs(self, attr_str: str) -> dict[str, str]:
        return {k: v for k, v in self._ATTR_RE.findall(attr_str)}

    def parse(self, ssml_text: str) -> list[Segment]:
        segments: list[Segment] = []
        cursor = 0

        for m in self._TAG_RE.finditer(ssml_text):
            # Text before this tag
            before = ssml_text[cursor:m.start()]
            if before.strip():
                segments.append(TextSegment(text=before.strip()))
            cursor = m.end()

            full_match = m.group(0)
            tag = (m.group('tag') or '').lower()
            attrs = self._parse_attrs(m.group('attrs') or '')
            inner = (m.group('inner') or '').strip()

            if tag == 'pause':
                ms = int(attrs.get('ms', 300))
                ms = max(50, min(ms, 5000))
                segments.append(SilenceSegment(duration_ms=ms))

            elif tag == 'break':
                strength = attrs.get('strength', 'sentence')
                ms = BREAK_STRENGTHS.get(strength, 350)
                if 'time' in attrs:
                    # time="500ms" or time="0.5s"
                    t = attrs['time']
                    if t.endswith('ms'):
                        ms = int(t[:-2])
                    elif t.endswith('s'):
                        ms = int(float(t[:-1]) * 1000)
                segments.append(SilenceSegment(duration_ms=ms))

            elif tag == 'emphasis':
                level = attrs.get('level', 'moderate')
                speed = EMPHASIS_SPEEDS.get(level, 1.0)
                if inner:
                    segments.append(TextSegment(text=inner, speed_override=speed))

            elif tag == 'say-as':
                interpret = attrs.get('interpret-as', '')
                if interpret == 'characters' and inner:
                    spaced = ' '.join(list(inner.upper()))
                    segments.append(TextSegment(text=spaced, speed_override=0.80))
                elif inner:
                    segments.append(TextSegment(text=inner))

            # Unrecognised tags: drop the tag, keep inner text if present
            elif inner:
                segments.append(TextSegment(text=inner))

        # Trailing text after last tag
        tail = ssml_text[cursor:].strip()
        if tail:
            segments.append(TextSegment(text=tail))

        return [s for s in segments if _segment_has_content(s)]


def _segment_has_content(seg: Segment) -> bool:
    if isinstance(seg, TextSegment):
        return bool(seg.text.strip())
    return seg.duration_ms > 0


# ── Assembler ─────────────────────────────────────────────────────────────────

def make_silence(duration_ms: int, sample_rate: int) -> np.ndarray:
    """Return a float32 silence array of the specified duration."""
    n_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(n_samples, dtype=np.float32)


def assemble_segments(
    synthesised: list[tuple[TextSegment, np.ndarray]],
    silence_segs: list[tuple[int, SilenceSegment]],
    all_segments: list[Segment],
    sample_rate: int,
) -> np.ndarray:
    """
    Merge synthesised text chunks and silence segments in original order.

    synthesised   : [(TextSegment, audio_array), ...] in segment order
    silence_segs  : [(original_index, SilenceSegment), ...]
    all_segments  : the full ordered list from SSMLParser
    sample_rate   : output sample rate
    """
    synth_map = {id(seg): audio for seg, audio in synthesised}
    silence_map = {id(seg): make_silence(seg.duration_ms, sample_rate) for _, seg in silence_segs}

    chunks: list[np.ndarray] = []
    for seg in all_segments:
        if isinstance(seg, TextSegment) and id(seg) in synth_map:
            chunks.append(synth_map[id(seg)])
        elif isinstance(seg, SilenceSegment) and id(seg) in silence_map:
            chunks.append(silence_map[id(seg)])

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def strip_ssml(text: str) -> str:
    """Remove all SSML tags from text — returns plain string."""
    return re.sub(r'<[^>]+>', '', text).strip()


# Module-level parser singleton
parser = SSMLParser()
