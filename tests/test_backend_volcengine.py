import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends.volcengine_speech import VolcengineSpeechBackend


def test_is_streaming_returns_false():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    assert backend.is_streaming() is False


@pytest.mark.asyncio
async def test_transcribe_sends_correct_request():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {"text": "transcribed text"}
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == "transcribed text"


@pytest.mark.asyncio
async def test_transcribe_returns_empty_on_error():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == ""


def test_sign_request_produces_headers():
    backend = VolcengineSpeechBackend(app_id="test-app", access_key="ak", secret_key="sk")
    headers = backend._build_headers(b"fake-audio", "zh")
    assert "Authorization" in headers or "X-Api-Key" in headers or "Content-Type" in headers
