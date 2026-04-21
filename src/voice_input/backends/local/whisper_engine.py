from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

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
        try:
            segments, _ = self._model.transcribe(
                audio_f32,
                language=language,
                beam_size=5,
                vad_filter=False,
            )
            text = "".join(seg.text for seg in segments)
            return text.strip()
        except Exception as e:
            log.error("Whisper transcription error: %s", e)
            return ""
