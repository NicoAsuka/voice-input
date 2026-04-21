from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TranscriptionBackend(ABC):
    """Abstract base class for all speech-to-text backends."""

    @abstractmethod
    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        """Transcribe int16 audio array, return text."""
        ...

    @abstractmethod
    def is_streaming(self) -> bool:
        """Whether this backend supports real-time incremental transcription."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Load model or validate API credentials."""
        ...

    async def cleanup(self) -> None:
        """Release resources. Default is no-op."""
        pass
