"""Unit tests for the SSML parser."""
import pytest
from backend.utils.ssml import SSMLParser, TextSegment, SilenceSegment, strip_ssml, make_silence
import numpy as np


parser = SSMLParser()


class TestSSMLParser:
    def test_plain_text_returns_single_text_segment(self):
        segs = parser.parse("Hello world")
        assert len(segs) == 1
        assert isinstance(segs[0], TextSegment)
        assert segs[0].text == "Hello world"

    def test_pause_tag(self):
        segs = parser.parse('Before<pause ms="300"/>After')
        assert len(segs) == 3
        assert isinstance(segs[1], SilenceSegment)
        assert segs[1].duration_ms == 300

    def test_pause_clamped_to_max(self):
        segs = parser.parse('<pause ms="9999"/>')
        assert segs[0].duration_ms == 5000

    def test_pause_clamped_to_min(self):
        segs = parser.parse('<pause ms="5"/>')
        assert segs[0].duration_ms == 50

    def test_break_paragraph(self):
        segs = parser.parse('Start<break strength="paragraph"/>End')
        assert any(isinstance(s, SilenceSegment) and s.duration_ms == 800 for s in segs)

    def test_break_sentence(self):
        segs = parser.parse('Start<break strength="sentence"/>End')
        assert any(isinstance(s, SilenceSegment) and s.duration_ms == 350 for s in segs)

    def test_emphasis_strong_speed(self):
        segs = parser.parse('<emphasis level="strong">critical word</emphasis>')
        assert len(segs) == 1
        assert isinstance(segs[0], TextSegment)
        assert abs(segs[0].speed_override - 0.85) < 0.01

    def test_emphasis_moderate_speed(self):
        segs = parser.parse('<emphasis level="moderate">word</emphasis>')
        assert abs(segs[0].speed_override - 0.92) < 0.01

    def test_emphasis_reduced_speed(self):
        segs = parser.parse('<emphasis level="reduced">word</emphasis>')
        assert abs(segs[0].speed_override - 1.10) < 0.01

    def test_say_as_characters(self):
        segs = parser.parse('<say-as interpret-as="characters">NASA</say-as>')
        assert len(segs) == 1
        assert isinstance(segs[0], TextSegment)
        assert "N" in segs[0].text and "A" in segs[0].text

    def test_complex_mixed_input(self):
        text = 'Good morning.<pause ms="500"/>Today we discuss <emphasis level="strong">climate change</emphasis><break strength="paragraph"/>Thank you.'
        segs = parser.parse(text)
        types = [type(s).__name__ for s in segs]
        assert "TextSegment" in types
        assert "SilenceSegment" in types
        # emphasis segment should have speed_override < 1.0
        emphasis_segs = [s for s in segs if isinstance(s, TextSegment) and s.speed_override < 1.0]
        assert len(emphasis_segs) == 1

    def test_empty_string_returns_empty(self):
        segs = parser.parse("")
        assert segs == []

    def test_only_whitespace_returns_empty(self):
        segs = parser.parse("   ")
        assert segs == []


class TestStripSSML:
    def test_strips_all_tags(self):
        result = strip_ssml('<pause ms="300"/>Hello <emphasis level="strong">world</emphasis>')
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result


class TestMakeSilence:
    def test_correct_length(self):
        silence = make_silence(500, sample_rate=44100)
        expected = int(44100 * 500 / 1000)
        assert len(silence) == expected
        assert np.all(silence == 0.0)

    def test_zero_duration(self):
        silence = make_silence(0, sample_rate=24000)
        assert len(silence) == 0
