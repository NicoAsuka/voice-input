import pytest
import numpy as np
from voice_input.backends.base import TranscriptionBackend


def test_cannot_instantiate_abc():
    """TranscriptionBackend is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        TranscriptionBackend()


def test_concrete_subclass_must_implement_all_methods():
    """A subclass missing abstract methods cannot be instantiated."""
    class Incomplete(TranscriptionBackend):
        pass

    with pytest.raises(TypeError):
        Incomplete()


@pytest.mark.asyncio
async def test_concrete_subclass_works():
    """A fully implemented subclass can be instantiated and called."""
    class Dummy(TranscriptionBackend):
        async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
            return "hello"

        def is_streaming(self) -> bool:
            return False

        async def initialize(self) -> None:
            pass

    backend = Dummy()
    assert backend.is_streaming() is False
    result = await backend.transcribe(np.zeros(100, dtype=np.int16), "zh")
    assert result == "hello"


@pytest.mark.asyncio
async def test_cleanup_default_is_noop():
    """Default cleanup() does nothing and doesn't raise."""
    class Dummy(TranscriptionBackend):
        async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
            return ""

        def is_streaming(self) -> bool:
            return False

        async def initialize(self) -> None:
            pass

    backend = Dummy()
    await backend.cleanup()  # should not raise
