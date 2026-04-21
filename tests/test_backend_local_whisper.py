import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from voice_input.backends.local_whisper import LocalWhisperBackend


def test_is_streaming_returns_true():
    backend = LocalWhisperBackend(model_name="tiny", language="zh", device="cpu")
    assert backend.is_streaming() is True


@pytest.mark.asyncio
async def test_initialize_loads_model():
    backend = LocalWhisperBackend(model_name="tiny", language="zh", device="cpu")
    mock_model = MagicMock()
    with patch("faster_whisper.WhisperModel", return_value=mock_model):
        await backend.initialize()
    assert backend._model is mock_model


@pytest.mark.asyncio
async def test_transcribe_returns_text():
    backend = LocalWhisperBackend(model_name="tiny", language="zh", device="cpu")
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)
    backend._model = mock_model

    audio = np.zeros(16000, dtype=np.int16)
    result = await backend.transcribe(audio, "zh")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_transcribe_short_audio_returns_empty():
    backend = LocalWhisperBackend(model_name="tiny", language="zh", device="cpu")
    backend._model = MagicMock()

    audio = np.zeros(100, dtype=np.int16)  # too short
    result = await backend.transcribe(audio, "zh")
    assert result == ""
