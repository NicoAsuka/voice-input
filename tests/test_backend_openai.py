import io
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from voice_input.backends.openai_whisper import OpenAIWhisperBackend


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


@pytest.mark.asyncio
async def test_cleanup_closes_client():
    backend = OpenAIWhisperBackend(api_base="https://api.openai.com/v1", api_key="sk-test", model="whisper-1")
    with patch.object(backend._client, "aclose", new_callable=AsyncMock) as mock_close:
        await backend.cleanup()
    mock_close.assert_called_once()
