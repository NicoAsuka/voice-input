import base64
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends.google_speech import GoogleSpeechBackend


def test_is_streaming_returns_false():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    assert backend.is_streaming() is False


def test_map_language():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    assert backend._map_language("zh") == "zh-CN"
    assert backend._map_language("en") == "en-US"
    assert backend._map_language("ja") == "ja-JP"
    assert backend._map_language("ko") == "ko-KR"
    assert backend._map_language("zh-TW") == "zh-TW"
    assert backend._map_language("fr") == "fr-FR"  # fallback


@pytest.mark.asyncio
async def test_transcribe_sends_correct_request():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    backend._access_token = "fake-token"

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"alternatives": [{"transcript": "hello"}]},
            {"alternatives": [{"transcript": " world"}]},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == "hello world"


@pytest.mark.asyncio
async def test_transcribe_returns_empty_on_no_results():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    backend._access_token = "fake-token"

    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "en")

    assert result == ""


@pytest.mark.asyncio
async def test_transcribe_returns_empty_on_error():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    backend._access_token = "fake-token"

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == ""
