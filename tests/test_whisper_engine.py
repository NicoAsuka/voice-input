import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_whisper_engine_is_streaming():
    from voice_input.backends.local.whisper_engine import WhisperEngine

    engine = WhisperEngine()
    assert engine.is_streaming() is True


def test_whisper_engine_load_model_cpu():
    from voice_input.backends.local.whisper_engine import WhisperEngine

    engine = WhisperEngine()
    mock_model = MagicMock()
    with patch("voice_input.backends.local.whisper_engine.WhisperModel", return_value=mock_model):
        engine.load_model("tiny", "cpu", Path("/tmp/models"))
    assert engine._model is mock_model


def test_whisper_engine_transcribe():
    from voice_input.backends.local.whisper_engine import WhisperEngine

    engine = WhisperEngine()
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)
    engine._model = mock_model

    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, "zh")
    assert result == "hello world"


def test_whisper_engine_transcribe_no_model_returns_empty():
    from voice_input.backends.local.whisper_engine import WhisperEngine

    engine = WhisperEngine()
    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, "zh")
    assert result == ""
