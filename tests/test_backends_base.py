from __future__ import annotations

import pytest

from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    RecognitionError,
    Session,
    TranscriptionBackend,
)


def test_recognition_error_has_user_message():
    e = RecognitionError("internal", user_message="识别失败")
    assert str(e) == "internal"
    assert e.user_message == "识别失败"


def test_recognition_error_default_user_message():
    e = RecognitionError("boom")
    assert e.user_message == "识别失败，请重试"


def test_backend_capabilities_defaults():
    caps = BackendCapabilities()
    assert caps.supports_streaming is False
    assert caps.requires_network is False
    assert caps.supports_vad is False


def test_backend_descriptor_fields():
    desc = BackendDescriptor(
        backend_id="fake",
        model_id="m1",
        capabilities=BackendCapabilities(supports_vad=True),
    )
    assert desc.backend_id == "fake"
    assert desc.model_id == "m1"
    assert desc.capabilities.supports_vad is True


def test_session_is_abstract():
    with pytest.raises(TypeError):
        Session()  # type: ignore[abstract]


def test_transcription_backend_is_abstract():
    with pytest.raises(TypeError):
        TranscriptionBackend()  # type: ignore[abstract]


def test_default_is_ready_returns_true():
    class Concrete(TranscriptionBackend):
        async def initialize(self): pass
        def describe(self): return BackendDescriptor("x", "y", BackendCapabilities())
        def create_session(self, language): raise NotImplementedError
        async def shutdown(self): pass
    assert Concrete().is_ready() is True
