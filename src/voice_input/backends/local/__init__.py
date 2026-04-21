"""Local STT backend with switchable engine (whisper / sensevoice)."""
from __future__ import annotations

import logging

import numpy as np

from voice_input.backends.base import TranscriptionBackend
from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine
from voice_input.backends.local.whisper_engine import WhisperEngine
from voice_input.config import AppConfig, xdg_cache_dir

log = logging.getLogger(__name__)

# Minimum samples to attempt transcription (0.1s at 16kHz)
MIN_SAMPLES = 1600


class LocalBackend(TranscriptionBackend):
    """Local transcription backend that delegates to a switchable engine."""

    def __init__(self, config: AppConfig) -> None:
        local_cfg = config.get("stt", {}).get("local", {})
        self._engine_name = local_cfg.get("engine", "whisper")
        self._model_name = local_cfg.get("model", "medium")
        self._device = local_cfg.get("device", "auto")
        self._engine = None

    def is_streaming(self) -> bool:
        if self._engine is None:
            return self._engine_name == "whisper"
        return self._engine.is_streaming()

    async def initialize(self) -> None:
        if self._engine_name == "whisper":
            self._engine = WhisperEngine()
        elif self._engine_name == "sensevoice":
            self._engine = SenseVoiceEngine()
        else:
            raise ValueError(f"Unknown local engine: {self._engine_name}")

        cache_dir = xdg_cache_dir() / "models"
        self._engine.load_model(self._model_name, self._device, cache_dir)

    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        if self._engine is None or len(audio_data) < MIN_SAMPLES:
            return ""
        audio_f32 = audio_data.astype(np.float32) / 32768.0
        return self._engine.transcribe(audio_f32, language)

    async def cleanup(self) -> None:
        self._engine = None
