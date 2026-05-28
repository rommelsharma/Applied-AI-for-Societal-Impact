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
                                              (Latin scripts only; non-Latin
                                               scripts are passed through as-is)

Tags are parsed into a list of Segment objects; the synthesiser walks the list
and assembles the final waveform from audio chunks and silence blocks.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
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

# Minimum non-whitespace characters for a text segment to be worth synthesising.
# Very short fragments (e.g. a lone punctuation mark left after tag extraction)
# produce silence or artefacts in most TTS engines.
MIN_SEGMENT_CHARS = 2


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


# ── Script detection helper ───────────────────────────────────────────────────

def _is_latin_script(text: str) -> bool:
    """
    Return True if the majority of letters in *text* are Latin-script characters.
    Used to decide whether letter-by-letter spelling makes linguistic sense.
    """
    latin_count = 0
    total_letters = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith('L'):          # any Unicode letter
            total_letters += 1
            name = unicodedata.name(ch, '')
            if name.startswith('LATIN') or name.startswith('SMALL LATIN') or ord(ch) < 0x0100:
                latin_count += 1
    if total_letters == 0:
        return True  # no letters at all — treat as Latin for safety
    return (latin_count / total_letters) >= 0.5


# ── Parser ────────────────────────────────────────────────────────────────────

class SSMLParser:
    """Parse an SSML-tagged string into a flat list of Segments.

    Design notes
    ------------
    * The inner-text capture uses [^<]+ rather than [^<]* so that empty wrapped
      tags (e.g. <emphasis></emphasis>) are not matched and left for the
      "unrecognised / empty" fallback path.
    * re.DOTALL is NOT used intentionally; newlines inside <emphasis> are
      captured by [^<]* because character classes are never affected by DOTALL.
    * Unicode text (Hindi, Japanese, Arabic, etc.) works transparently because
      Python's re engine handles Unicode by default.
    """

    # Two independent top-level alternatives:
    #   1. Any handled or unhandled tag (self-closing or wrapping)
    #   2. <break> in its various forms (backup for non-self-closing variants)
    _TAG_RE = re.compile(
        # Alt 1 ── generic tag: self-closing or wrapping inner text
        r'<(?P<tag>[a-zA-Z][a-zA-Z0-9\-]*)'   # opening tag name (ASCII)
        r'(?P<attrs>[^>]*?)'                    # attributes (non-greedy, no closing >)
        r'(?:'
            r'/>'                               # self-closing: />
            r'|>(?P<inner>[^<]*?)</[a-zA-Z][a-zA-Z0-9\-]*\s*>'  # wrapping: >…</tag>
        r')'
        r'|'
        # Alt 2 ── <break> shorthand (non-self-closing, no inner content)
        r'<break(?P<break_attrs>[^>]*)>',
        re.IGNORECASE,
    )

    _ATTR_RE = re.compile(r'([\w][\w\-]*)=["\']([^"\']*)["\']')

    def _parse_attrs(self, attr_str: str) -> dict[str, str]:
        return {k: v for k, v in self._ATTR_RE.findall(attr_str or '')}

    def parse(self, ssml_text: str) -> list[Segment]:
        segments: list[Segment] = []
        cursor = 0

        for m in self._TAG_RE.finditer(ssml_text):
            # ── Text before this tag ──────────────────────────────────────
            before = ssml_text[cursor:m.start()]
            if before.strip():
                segments.append(TextSegment(text=before.strip()))
            cursor = m.end()

            tag = (m.group('tag') or '').lower()
            attrs = self._parse_attrs(m.group('attrs') or '')
            inner = (m.group('inner') or '').strip()

            # Alt 2 match: <break ...> (no named 'tag' group matched)
            if not tag:
                break_attrs = self._parse_attrs(m.group('break_attrs') or '')
                strength = break_attrs.get('strength', 'sentence')
                ms = BREAK_STRENGTHS.get(strength, 350)
                if 'time' in break_attrs:
                    ms = _parse_time(break_attrs['time'])
                segments.append(SilenceSegment(duration_ms=ms))
                continue

            if tag == 'pause':
                ms = int(attrs.get('ms', 300))
                ms = max(50, min(ms, 5000))
                segments.append(SilenceSegment(duration_ms=ms))

            elif tag == 'break':
                strength = attrs.get('strength', 'sentence')
                ms = BREAK_STRENGTHS.get(strength, 350)
                if 'time' in attrs:
                    ms = _parse_time(attrs['time'])
                segments.append(SilenceSegment(duration_ms=ms))

            elif tag == 'emphasis':
                level = attrs.get('level', 'moderate')
                speed = EMPHASIS_SPEEDS.get(level, 1.0)
                if inner:
                    segments.append(TextSegment(text=inner, speed_override=speed))

            elif tag == 'say-as':
                interpret = attrs.get('interpret-as', '')
                if interpret == 'characters' and inner:
                    if _is_latin_script(inner):
                        # Spell out Latin characters with spaces between each
                        spaced = ' '.join(list(inner.upper()))
                        segments.append(TextSegment(text=spaced, speed_override=0.80))
                    else:
                        # Non-Latin scripts (Hindi, Japanese, Arabic, etc.):
                        # letter-by-letter pronunciation doesn't map to written
                        # characters — pass the text through as-is so the TTS
                        # engine handles it with its own grapheme-to-phoneme rules.
                        logger.debug(
                            "say-as interpret-as='characters' with non-Latin text — "
                            "passing through: %r", inner
                        )
                        segments.append(TextSegment(text=inner))
                elif inner:
                    segments.append(TextSegment(text=inner))

            # Unrecognised tags: drop tag markup, preserve inner text
            elif inner:
                segments.append(TextSegment(text=inner))

        # Trailing text after the last tag
        tail = ssml_text[cursor:].strip()
        if tail:
            segments.append(TextSegment(text=tail))

        return [s for s in segments if _segment_has_content(s)]


def _parse_time(time_str: str) -> int:
    """Parse SSML time attribute like '500ms' or '0.5s' → milliseconds."""
    time_str = time_str.strip()
    if time_str.endswith('ms'):
        try:
            return int(time_str[:-2])
        except ValueError:
            return 350
    elif time_str.endswith('s'):
        try:
            return int(float(time_str[:-1]) * 1000)
        except ValueError:
            return 350
    return 350


def _segment_has_content(seg: Segment) -> bool:
    if isinstance(seg, TextSegment):
        # Require at least MIN_SEGMENT_CHARS non-whitespace characters so
        # lone punctuation marks or stray spaces don't trigger synthesis calls
        return len(seg.text.replace(' ', '').replace('\n', '').replace('\t', '')) >= MIN_SEGMENT_CHARS
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
