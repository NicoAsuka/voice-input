import numpy as np
from pathlib import Path
from voice_input.backends.local.engine import LocalEngine


def test_protocol_conformance_with_complete_class():
    """A class implementing all protocol methods is recognized."""
    class FakeEngine:
        def load_model(self, model_name: str, device: str, cache_dir: Path) -> None:
            pass

        def transcribe(self, audio_f32: np.ndarray, language: str) -> str:
            return ""

        def is_streaming(self) -> bool:
            return False

    engine: LocalEngine = FakeEngine()
    assert engine.is_streaming() is False
    assert engine.transcribe(np.zeros(100, dtype=np.float32), "zh") == ""
