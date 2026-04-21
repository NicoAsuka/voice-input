"""STT backend abstraction and factory."""
from __future__ import annotations

from voice_input.backends.base import TranscriptionBackend
from voice_input.config import AppConfig


def create_backend(config: AppConfig) -> TranscriptionBackend:
    """Create the appropriate STT backend based on config."""
    backend_name = config.get("stt", {}).get("backend", "local")
    if backend_name == "local":
        from voice_input.backends.local import LocalBackend

        return LocalBackend(config)
    raise ValueError(f"Unknown STT backend: {backend_name}")
