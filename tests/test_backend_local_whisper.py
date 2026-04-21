"""Tests for LocalBackend with whisper engine."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _whisper_config():
    return {
        "stt": {
            "backend": "local",
            "local": {
                "engine": "whisper",
                "model": "tiny",
                "language": "zh",
                "device": "cpu",
            },
        },
    }


def test_is_streaming_returns_true():
    from voice_input.backends.local import LocalBackend

    backend = LocalBackend(_whisper_config())
    assert backend.is_streaming() is True


@pytest.mark.asyncio
async def test_initialize_loads_model():
    from voice_input.backends.local import LocalBackend

    backend = LocalBackend(_whisper_config())
    mock_model = MagicMock()
    with patch("voice_input.backends.local.whisper_engine.WhisperModel", return_value=mock_model):
        await backend.initialize()
    assert backend._engine is not None
    assert backend._engine._model is mock_model


@pytest.mark.asyncio
async def test_transcribe_returns_text():
    from voice_input.backends.local import LocalBackend

    backend = LocalBackend(_whisper_config())
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)

    with patch("voice_input.backends.local.whisper_engine.WhisperModel", return_value=mock_model):
        await backend.initialize()
    audio = np.ones(16000, dtype=np.int16) * 1000
    result = await backend.transcribe(audio, "zh")
    assert result == "hello world"
    kwargs = mock_model.transcribe.call_args.kwargs
    assert kwargs["vad_filter"] is True
    assert kwargs["condition_on_previous_text"] is False


@pytest.mark.asyncio
async def test_transcribe_silent_audio_returns_empty_without_model_call():
    from voice_input.backends.local import LocalBackend

    backend = LocalBackend(_whisper_config())
    mock_model = MagicMock()
    with patch("voice_input.backends.local.whisper_engine.WhisperModel", return_value=mock_model):
        await backend.initialize()

    audio = np.zeros(16000, dtype=np.int16)
    result = await backend.transcribe(audio, "zh")
    assert result == ""
    mock_model.transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_transcribe_short_audio_returns_empty():
    from voice_input.backends.local import LocalBackend

    backend = LocalBackend(_whisper_config())
    backend._engine = MagicMock()

    audio = np.zeros(100, dtype=np.int16)
    result = await backend.transcribe(audio, "zh")
    assert result == ""
