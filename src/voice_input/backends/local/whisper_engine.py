from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MIN_RMS = 32.0 / 32768.0
MIN_PEAK = 256.0 / 32768.0

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]


class WhisperEngine:
    """Local faster-whisper engine implementation."""

    def __init__(self) -> None:
        self._model = None

    def is_streaming(self) -> bool:
        return True

    def load_model(self, model_name: str, device: str, cache_dir: Path) -> None:
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install with: pip install voice-input[whisper]"
            )

        compute_type = "int8"
        actual_device = "cpu"
        if device in ("auto", "cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    actual_device = "cuda"
                    compute_type = "float16"
            except ImportError:
                pass

        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "Loading whisper model=%s device=%s compute_type=%s",
            model_name,
            actual_device,
            compute_type,
        )
        self._model = WhisperModel(
            model_name,
            device=actual_device,
            compute_type=compute_type,
            download_root=str(cache_dir),
        )
        log.info("Whisper model loaded and ready")

    def transcribe(self, audio_f32: np.ndarray, language: str) -> str:
        if self._model is None:
            return ""
        rms = float(np.sqrt(np.mean(audio_f32.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(audio_f32))) if len(audio_f32) else 0.0
        if rms < MIN_RMS or peak < MIN_PEAK:
            log.warning(
                "Local audio below speech threshold; skipping transcription "
                "(rms=%.4f peak=%.4f)",
                rms,
                peak,
            )
            return ""
        try:
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
            log.error("Whisper transcription error: %s", e)
            return ""
