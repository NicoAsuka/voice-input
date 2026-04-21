import pytest
from unittest.mock import patch

from voice_input.backends import create_backend
from voice_input.backends.base import TranscriptionBackend
from voice_input.backends.google_speech import GoogleSpeechBackend
from voice_input.backends.local_whisper import LocalWhisperBackend
from voice_input.backends.openai_whisper import OpenAIWhisperBackend
from voice_input.backends.volcengine_speech import VolcengineSpeechBackend


def _make_config(backend: str = "local", **overrides) -> dict:
    cfg = {
        "whisper": {"model": "tiny", "language": "zh", "device": "cpu"},
        "stt": {
            "backend": backend,
            "openai": {"api_base": "https://api.openai.com/v1", "model": "whisper-1"},
            "google": {"credentials_path": ""},
            "volcengine": {"app_id": "test"},
        },
    }
    cfg.update(overrides)
    return cfg


def test_create_local_backend():
    backend = create_backend(_make_config("local"))
    assert isinstance(backend, LocalWhisperBackend)


def test_create_openai_backend():
    with patch("keyring.get_password", return_value="sk-test"):
        backend = create_backend(_make_config("openai"))
    assert isinstance(backend, OpenAIWhisperBackend)


def test_create_google_backend():
    backend = create_backend(_make_config("google"))
    assert isinstance(backend, GoogleSpeechBackend)


def test_create_volcengine_backend():
    with patch("keyring.get_password", return_value="fake-key"):
        backend = create_backend(_make_config("volcengine"))
    assert isinstance(backend, VolcengineSpeechBackend)


def test_create_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown STT backend"):
        create_backend(_make_config("nonexistent"))


def test_all_backends_are_transcription_backends():
    local = create_backend(_make_config("local"))
    assert isinstance(local, TranscriptionBackend)

    with patch("keyring.get_password", return_value="sk-test"):
        openai_b = create_backend(_make_config("openai"))
    assert isinstance(openai_b, TranscriptionBackend)
