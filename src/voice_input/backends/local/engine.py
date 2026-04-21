from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LocalEngine(Protocol):
    """Protocol for local STT engine implementations."""

    def load_model(self, model_name: str, device: str, cache_dir: Path) -> None:
        """Load model into memory."""
        ...

    def transcribe(self, audio_f32: np.ndarray, language: str) -> str:
        """Transcribe float32 audio array, return text."""
        ...

    def is_streaming(self) -> bool:
        """Whether this engine supports real-time incremental transcription."""
        ...
