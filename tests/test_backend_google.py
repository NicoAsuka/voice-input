import base64
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends.base import BackendCapabilities, BackendDescriptor, RecognitionError
from voice_input.backends.google_speech import GoogleSpeechBackend, _GoogleSession


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


# --- New Session interface tests ---


def test_is_ready_with_credentials():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    assert backend.is_ready() is True


def test_is_ready_without_credentials():
    backend = GoogleSpeechBackend(credentials_path="")
    backend.credentials_path = ""
    assert backend.is_ready() is False


def test_describe_returns_descriptor():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    desc = backend.describe()
    assert isinstance(desc, BackendDescriptor)
    assert desc.backend_id == "google"
    assert desc.model_id == "google-stt-default"
    assert desc.capabilities.supports_streaming is False
    assert desc.capabilities.requires_network is True


def test_create_session_returns_session():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    session = backend.create_session("en")
    assert isinstance(session, _GoogleSession)
    assert session._language == "en"


@pytest.mark.asyncio
async def test_shutdown_closes_client():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    with patch.object(backend._client, "aclose", new_callable=AsyncMock) as mock_close:
        await backend.shutdown()
    mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_session_finish_returns_text():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    session = _GoogleSession(backend, "zh")
    audio = np.zeros(16000, dtype=np.int16)
    session.push_audio(audio)

    with patch.object(backend, "transcribe", new_callable=AsyncMock, return_value="hello"):
        result = await session.finish()
    assert result == "hello"


@pytest.mark.asyncio
async def test_session_finish_concatenates_chunks():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    session = _GoogleSession(backend, "zh")
    session.push_audio(np.zeros(100, dtype=np.int16))
    session.push_audio(np.zeros(200, dtype=np.int16))

    with patch.object(backend, "transcribe", new_callable=AsyncMock, return_value="ok") as mock_t:
        await session.finish()
    assert mock_t.call_args[0][0].shape == (300,)


@pytest.mark.asyncio
async def test_session_finish_empty_returns_empty():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    session = _GoogleSession(backend, "zh")
    result = await session.finish()
    assert result == ""


@pytest.mark.asyncio
async def test_session_cancel_returns_empty():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    session = _GoogleSession(backend, "zh")
    session.push_audio(np.zeros(100, dtype=np.int16))
    session.cancel()
    assert session._cancelled is True
    assert len(session._buffer) == 0
    result = await session.finish()
    assert result == ""


@pytest.mark.asyncio
async def test_session_finish_raises_recognition_error():
    backend = GoogleSpeechBackend(credentials_path="/fake/creds.json")
    session = _GoogleSession(backend, "zh")
    session.push_audio(np.zeros(100, dtype=np.int16))

    with patch.object(backend, "transcribe", new_callable=AsyncMock, side_effect=Exception("api down")):
        with pytest.raises(RecognitionError) as exc_info:
            await session.finish()
    assert "Google STT error" in exc_info.value.user_message
