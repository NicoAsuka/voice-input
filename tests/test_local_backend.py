import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_local_backend_initializes_whisper_engine():
    from voice_input.backends.local import LocalBackend

    config = {
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
    backend = LocalBackend(config)
    mock_engine = MagicMock()
    with patch("voice_input.backends.local.WhisperEngine", return_value=mock_engine):
        await backend.initialize()
    mock_engine.load_model.assert_called_once()
    assert backend.is_streaming() == mock_engine.is_streaming()


@pytest.mark.asyncio
async def test_local_backend_initializes_sensevoice_engine():
    from voice_input.backends.local import LocalBackend

    config = {
        "stt": {
            "backend": "local",
            "local": {
                "engine": "sensevoice",
                "model": "iic/SenseVoiceSmall",
                "language": "zh",
                "device": "cpu",
            },
        },
    }
    backend = LocalBackend(config)
    mock_engine = MagicMock()
    with patch("voice_input.backends.local.SenseVoiceEngine", return_value=mock_engine):
        await backend.initialize()
    mock_engine.load_model.assert_called_once()


@pytest.mark.asyncio
async def test_local_backend_uses_sensevoice_default_when_model_is_whisper_name():
    from voice_input.backends.local import LocalBackend

    config = {
        "stt": {
            "backend": "local",
            "local": {
                "engine": "sensevoice",
                "model": "medium",
                "language": "zh",
                "device": "cpu",
            },
        },
    }
    backend = LocalBackend(config)
    mock_engine = MagicMock()
    with patch("voice_input.backends.local.SenseVoiceEngine", return_value=mock_engine):
        await backend.initialize()
    assert mock_engine.load_model.call_args.args[0] == "iic/SenseVoiceSmall"


@pytest.mark.asyncio
async def test_local_backend_unknown_engine_raises():
    from voice_input.backends.local import LocalBackend

    config = {
        "stt": {
            "backend": "local",
            "local": {
                "engine": "unknown",
                "model": "x",
                "language": "zh",
                "device": "cpu",
            },
        },
    }
    backend = LocalBackend(config)
    with pytest.raises(ValueError, match="Unknown local engine"):
        await backend.initialize()


@pytest.mark.asyncio
async def test_local_backend_transcribe_delegates():
    from voice_input.backends.local import LocalBackend

    config = {
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
    backend = LocalBackend(config)
    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = "hello"
    backend._engine = mock_engine

    audio = np.zeros(16000, dtype=np.int16)
    result = await backend.transcribe(audio, "zh")
    assert result == "hello"
