"""API integration tests — run: pytest tests/ -v"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "kokoro_ready" in data
    assert "xtts_ready" in data


def test_voices_returns_list():
    response = client.get("/voices")
    assert response.status_code == 200
    voices = response.json()
    assert isinstance(voices, list)
    assert len(voices) > 0
    for v in voices:
        assert "id" in v
        assert "name" in v
        assert "engine" in v


def test_synthesize_missing_text():
    response = client.post("/synthesize", json={"voice": "af_bella", "text": ""})
    assert response.status_code == 422  # pydantic min_length validation


def test_synthesize_unknown_voice():
    with patch("backend.api.routes._kokoro") as mock_kokoro:
        mock_kokoro.is_ready.return_value = True
        mock_kokoro.synthesize.side_effect = ValueError("Unknown Kokoro voice: invalid_voice")
        response = client.post(
            "/synthesize",
            json={"text": "Hello world", "voice": "invalid_voice", "language": "en-us"},
        )
    assert response.status_code == 400


def test_synthesize_kokoro_success():
    import io
    import numpy as np
    import soundfile as sf

    dummy_audio = np.zeros(24000, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, dummy_audio, 24000, format="WAV")
    wav_bytes = buf.getvalue()

    with patch("backend.api.routes._kokoro") as mock_kokoro:
        mock_kokoro.is_ready.return_value = True
        mock_kokoro.synthesize.return_value = wav_bytes
        response = client.post(
            "/synthesize",
            json={"text": "Hello world", "voice": "af_bella", "language": "en-us"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "audio_url" in data
    assert data["engine"] == "kokoro"


def test_upload_invalid_file_type():
    response = client.post(
        "/upload-voice-sample",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_list_voice_samples():
    response = client.get("/voice-samples")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
