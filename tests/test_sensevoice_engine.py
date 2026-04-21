import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_sensevoice_engine_is_streaming():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine

    engine = SenseVoiceEngine()
    assert engine.is_streaming() is False


def test_sensevoice_engine_load_model():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine

    engine = SenseVoiceEngine()
    mock_model = MagicMock()
    with patch("voice_input.backends.local.sensevoice_engine.AutoModel", mock_model):
        engine.load_model("iic/SenseVoiceSmall", "cpu", Path("/tmp/models"))
    mock_model.assert_called_once_with(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="cpu",
        cache_dir="/tmp/models",
    )


def test_sensevoice_engine_load_model_default_name():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine

    engine = SenseVoiceEngine()
    mock_model = MagicMock()
    with patch("voice_input.backends.local.sensevoice_engine.AutoModel", mock_model):
        engine.load_model("", "cpu", Path("/tmp/models"))
    mock_model.assert_called_once_with(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="cpu",
        cache_dir="/tmp/models",
    )


def test_sensevoice_engine_load_model_missing_funasr():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine

    engine = SenseVoiceEngine()
    with patch("voice_input.backends.local.sensevoice_engine.AutoModel", None):
        with pytest.raises(RuntimeError, match="funasr"):
            engine.load_model("iic/SenseVoiceSmall", "cpu", Path("/tmp/models"))


def test_sensevoice_engine_transcribe():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine

    engine = SenseVoiceEngine()
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"text": "你好世界"}]
    engine._model = mock_model

    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, "zh")
    assert result == "你好世界"
    mock_model.generate.assert_called_once_with(
        input=audio,
        language="zh",
        use_itn=True,
    )


def test_sensevoice_engine_transcribe_empty_result():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine

    engine = SenseVoiceEngine()
    mock_model = MagicMock()
    mock_model.generate.return_value = []
    engine._model = mock_model

    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, "zh")
    assert result == ""


def test_sensevoice_engine_transcribe_no_model():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine

    engine = SenseVoiceEngine()
    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, "zh")
    assert result == ""
