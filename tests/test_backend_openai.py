import io
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends.base import BackendCapabilities, BackendDescriptor, RecognitionError
from voice_input.backends.openai_whisper import OpenAIWhisperBackend, _OpenAISession


def test_is_streaming_returns_false():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    assert backend.is_streaming() is False


def test_encode_wav_produces_valid_wav():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    audio = np.array([0, 100, -100, 32767, -32768], dtype=np.int16)
    wav_bytes = backend._encode_wav(audio)
    # WAV header starts with RIFF
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


@pytest.mark.asyncio
async def test_transcribe_sends_correct_request():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")

    mock_response = MagicMock()
    mock_response.json.return_value = {"text": "transcribed text"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == "transcribed text"
    call_kwargs = mock_client.post.call_args
    assert call_kwargs[0][0] == "/audio/transcriptions"


@pytest.mark.asyncio
async def test_transcribe_returns_empty_on_error():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")

    with patch.object(backend, "_client") as mock_client:
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        audio = np.zeros(16000, dtype=np.int16)
        result = await backend.transcribe(audio, "zh")

    assert result == ""



# --- New Session interface tests ---


def test_is_ready_with_client():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    assert backend.is_ready() is True


def test_is_ready_without_client():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    backend._client = None
    assert backend.is_ready() is False


def test_describe_returns_descriptor():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    desc = backend.describe()
    assert isinstance(desc, BackendDescriptor)
    assert desc.backend_id == "openai-whisper"
    assert desc.model_id == "whisper-1"
    assert desc.capabilities.supports_streaming is False
    assert desc.capabilities.requires_network is True


def test_create_session_returns_session():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    session = backend.create_session("zh")
    assert isinstance(session, _OpenAISession)
    assert session._language == "zh"


@pytest.mark.asyncio
async def test_shutdown_closes_client():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    with patch.object(backend._client, "aclose", new_callable=AsyncMock) as mock_close:
        await backend.shutdown()
    mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_session_finish_returns_text():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    session = _OpenAISession(backend, "zh")
    audio = np.zeros(16000, dtype=np.int16)
    session.push_audio(audio)
    assert len(session._buffer) == 1

    with patch.object(backend, "transcribe", new_callable=AsyncMock, return_value="hello"):
        result = await session.finish()
    assert result == "hello"


@pytest.mark.asyncio
async def test_session_finish_concatenates_chunks():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    session = _OpenAISession(backend, "zh")
    session.push_audio(np.zeros(100, dtype=np.int16))
    session.push_audio(np.zeros(200, dtype=np.int16))

    with patch.object(backend, "transcribe", new_callable=AsyncMock, return_value="ok") as mock_t:
        await session.finish()
    assert mock_t.call_args[0][0].shape == (300,)


@pytest.mark.asyncio
async def test_session_finish_empty_returns_empty():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    session = _OpenAISession(backend, "zh")
    result = await session.finish()
    assert result == ""


@pytest.mark.asyncio
async def test_session_cancel_returns_empty():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    session = _OpenAISession(backend, "zh")
    session.push_audio(np.zeros(100, dtype=np.int16))
    session.cancel()
    assert session._cancelled is True
    assert len(session._buffer) == 0
    result = await session.finish()
    assert result == ""


@pytest.mark.asyncio
async def test_session_finish_raises_recognition_error():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    session = _OpenAISession(backend, "zh")
    session.push_audio(np.zeros(100, dtype=np.int16))

    with patch.object(backend, "transcribe", new_callable=AsyncMock, side_effect=Exception("api down")):
        with pytest.raises(RecognitionError) as exc_info:
            await session.finish()
    assert "OpenAI Whisper error" in exc_info.value.user_message
