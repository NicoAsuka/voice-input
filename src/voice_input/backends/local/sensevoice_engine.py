from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:
    from funasr import AutoModel
except ImportError:
    AutoModel = None  # type: ignore[assignment,misc]

DEFAULT_MODEL = "iic/SenseVoiceSmall"


class SenseVoiceEngine:
    """Local SenseVoice Small engine using funasr."""

    def __init__(self) -> None:
        self._model = None

    def is_streaming(self) -> bool:
        return False

    def load_model(self, model_name: str, device: str, cache_dir: Path) -> None:
        if AutoModel is None:
            raise RuntimeError(
                "funasr is not installed. "
                "Install with: pip install voice-input[sensevoice]"
            )

        actual_device = self._resolve_device(device)
        resolved_name = model_name or DEFAULT_MODEL
        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "Loading SenseVoice model=%s device=%s",
            resolved_name,
            actual_device,
        )
        self._model = AutoModel(
            model=resolved_name,
            trust_remote_code=True,
            device=actual_device,
            cache_dir=str(cache_dir),
        )
        log.info("SenseVoice model loaded and ready")

    def transcribe(self, audio_f32: np.ndarray, language: str) -> str:
        if self._model is None:
            return ""
        try:
            result = self._model.generate(
                input=audio_f32,
                language=language,
                use_itn=True,
            )
            if result and len(result) > 0:
                return result[0].get("text", "").strip()
            return ""
        except Exception as e:
            log.error("SenseVoice transcription error: %s", e)
            return ""

    def _resolve_device(self, device: str) -> str:
        if device in ("auto", "cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda:0"
            except ImportError:
                pass
        return "cpu"
