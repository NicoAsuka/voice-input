from __future__ import annotations

import logging

import numpy as np

from voice_input.backends.base import TranscriptionBackend
from voice_input.config import xdg_cache_dir

log = logging.getLogger(__name__)

# Minimum samples to attempt transcription (0.1s at 16kHz)
MIN_SAMPLES = 1600
MIN_RMS = 32.0
MIN_PEAK = 256

# Try to import WhisperModel at module level for proper mocking in tests
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]


class LocalWhisperBackend(TranscriptionBackend):
    """Local faster-whisper transcription backend."""

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def is_streaming(self) -> bool:
        return True

    async def initialize(self) -> None:
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed")
        if self._model is not None:
            log.debug("Model already loaded, skipping re-initialization")
            return
        try:
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
        except Exception as e:
            log.error("Failed to load whisper model: %s", e)
            raise

    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        if self._model is None or len(audio_data) < MIN_SAMPLES:
            return ""
        rms = float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))
        peak = int(np.max(np.abs(audio_data))) if len(audio_data) else 0
        duration = len(audio_data) / 16000.0
        log.info(
            "Local audio stats: duration=%.2fs rms=%.1f peak=%d",
            duration,
            rms,
            peak,
        )
        if rms < MIN_RMS or peak < MIN_PEAK:
            log.warning(
                "Local audio below speech threshold; skipping transcription "
                "(rms=%.1f peak=%d)",
                rms,
                peak,
            )
            return ""
        try:
            audio_f32 = audio_data.astype(np.float32) / 32768.0
            segments, _ = self._model.transcribe(
                audio_f32,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
                hallucination_silence_threshold=1.0,
            )
            text = "".join(seg.text for seg in segments)
            return text.strip()
        except Exception as e:
            log.error("Transcription error: %s", e)
            return ""
