from __future__ import annotations

import logging

import numpy as np

from voice_input.backends.base import TranscriptionBackend
from voice_input.config import xdg_cache_dir

log = logging.getLogger(__name__)

# Minimum samples to attempt transcription (0.1s at 16kHz)
MIN_SAMPLES = 1600


class LocalWhisperBackend(TranscriptionBackend):
    """Local faster-whisper transcription backend."""

    def __init__(
        self,
        model_name: str = "medium",
        language: str = "zh",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device
        self._model = None

    def is_streaming(self) -> bool:
        return True

    async def initialize(self) -> None:
        from faster_whisper import WhisperModel

        model_dir = xdg_cache_dir() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        compute_type = "int8"
        actual_device = "cpu"
        if self.device in ("auto", "cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    actual_device = "cuda"
                    compute_type = "float16"
            except ImportError:
                pass

        log.info(
            "Loading whisper model=%s device=%s compute_type=%s",
            self.model_name, actual_device, compute_type,
        )
        self._model = WhisperModel(
            self.model_name,
            device=actual_device,
            compute_type=compute_type,
            download_root=str(model_dir),
        )
        log.info("Whisper model loaded and ready")

    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        if self._model is None or len(audio_data) < MIN_SAMPLES:
            return ""
        try:
            audio_f32 = audio_data.astype(np.float32) / 32768.0
            segments, _ = self._model.transcribe(
                audio_f32,
                language=language,
                beam_size=5,
                vad_filter=False,
            )
            text = "".join(seg.text for seg in segments)
            return text.strip()
        except Exception as e:
            log.error("Transcription error: %s", e)
            return ""
