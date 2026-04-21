"""Integration tests for STT backend switching."""
import copy
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from voice_input.backends import create_backend
from voice_input.backends.base import TranscriptionBackend
from voice_input.config import DEFAULT_CONFIG


def test_default_config_creates_local_backend():
    """Default config should create a local backend."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    backend = create_backend(cfg)
    assert backend.is_streaming() is True


def test_all_backends_implement_interface():
    """Every backend returned by the factory implements TranscriptionBackend."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    cfg["stt"]["backend"] = "local"
    local = create_backend(cfg)
    assert isinstance(local, TranscriptionBackend)
    assert local.is_streaming() is True

    cfg["stt"]["backend"] = "openai"
    with patch("keyring.get_password", return_value="sk-test"):
        openai_b = create_backend(cfg)
    assert isinstance(openai_b, TranscriptionBackend)
    assert openai_b.is_streaming() is False

    cfg["stt"]["backend"] = "google"
    google_b = create_backend(cfg)
    assert isinstance(google_b, TranscriptionBackend)
    assert google_b.is_streaming() is False

    cfg["stt"]["backend"] = "volcengine"
    with patch("keyring.get_password", return_value="fake"):
        volc_b = create_backend(cfg)
    assert isinstance(volc_b, TranscriptionBackend)
    assert volc_b.is_streaming() is False


@pytest.mark.asyncio
async def test_remote_backend_transcribe_flow():
    """Simulate remote backend: record, buffer, transcribe."""
    from voice_input.backends.openai_whisper import OpenAIWhisperBackend

    backend = OpenAIWhisperBackend(
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
    )

    audio_buffer = np.zeros(32000, dtype=np.int16)
    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "hello world"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        result = await backend.transcribe(audio_buffer, "en")

    assert result == "hello world"
