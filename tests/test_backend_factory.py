from __future__ import annotations

import pytest
from unittest.mock import patch

from voice_input.backends import create_backend
from voice_input.backends.sherpa_backend import SherpaBackend


def test_create_sherpa_backend():
    cfg = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "x"}}}
    backend = create_backend(cfg)
    assert isinstance(backend, SherpaBackend)


def test_create_unknown_backend_raises():
    cfg = {"stt": {"backend": "nonexistent"}}
    with pytest.raises(ValueError, match="Unknown STT backend"):
        create_backend(cfg)


def test_create_local_backend_id_now_routes_to_sherpa():
    """Legacy 'local' value still works (treated as sherpa)."""
    cfg = {"stt": {"backend": "local", "sherpa": {}}}
    backend = create_backend(cfg)
    assert isinstance(backend, SherpaBackend)


def test_create_volcengine_backend(monkeypatch):
    monkeypatch.setattr(
        "keyring.get_password",
        lambda service, key: "fake-key" if "volcengine" in key else None,
    )
    from voice_input.backends.volcengine_speech import VolcengineSpeechBackend
    cfg = {"stt": {"backend": "volcengine", "volcengine": {"app_id": "x", "resource_id": "r"}}}
    backend = create_backend(cfg)
    assert isinstance(backend, VolcengineSpeechBackend)


def test_create_openai_backend(monkeypatch):
    monkeypatch.setattr(
        "keyring.get_password",
        lambda service, key: "sk-test",
    )
    from voice_input.backends.openai_whisper import OpenAIWhisperBackend
    cfg = {"stt": {"backend": "openai", "openai": {}}}
    backend = create_backend(cfg)
    assert isinstance(backend, OpenAIWhisperBackend)


def test_create_google_backend():
    from voice_input.backends.google_speech import GoogleSpeechBackend
    cfg = {"stt": {"backend": "google", "google": {"credentials_path": "/tmp/creds.json"}}}
    backend = create_backend(cfg)
    assert isinstance(backend, GoogleSpeechBackend)


def test_default_backend_is_sherpa():
    cfg = {"stt": {}}
    backend = create_backend(cfg)
    assert isinstance(backend, SherpaBackend)
