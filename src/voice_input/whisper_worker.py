# src/voice_input/whisper_worker.py
from __future__ import annotations

import logging
import queue
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from voice_input.config import xdg_cache_dir

log = logging.getLogger(__name__)


class WhisperWorker(QThread):
    """Runs faster-whisper transcription in a background thread.

    Every 0.5s, drains the audio queue, accumulates into a buffer,
    and runs transcription on the full buffer.
    """

    transcription_updated = pyqtSignal(str)
    model_ready = pyqtSignal()
    error_occurred = pyqtSignal(str)

    POLL_INTERVAL_MS = 500

    def __init__(
        self,
        whisper_queue: queue.Queue,
        model_name: str = "medium",
        language: str = "zh",
        device: str = "auto",
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.whisper_queue = whisper_queue
        self.model_name = model_name
        self.language = language
        self.device = device
        self.audio_buffer = np.array([], dtype=np.int16)
        self._model = None
        self._running = False

    def drain_queue(self) -> np.ndarray:
        """Drain all available chunks from the queue. Returns concatenated int16 array."""
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self.whisper_queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.array([], dtype=np.int16)
        return np.concatenate(chunks)

    def accumulate(self) -> None:
        """Drain queue and append to the cumulative audio buffer."""
        new_data = self.drain_queue()
        if len(new_data) > 0:
            self.audio_buffer = np.concatenate([self.audio_buffer, new_data])

    def reset(self) -> None:
        """Clear the audio buffer for a new recording session."""
        self.audio_buffer = np.array([], dtype=np.int16)
        # Also drain any leftover audio
        self.drain_queue()

    def _load_model(self) -> bool:
        """Load the faster-whisper model. Returns True on success."""
        try:
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
            self.model_ready.emit()
            return True
        except Exception as e:
            log.error("Failed to load whisper model: %s", e)
            self.error_occurred.emit(f"Whisper model load failed: {e}")
            return False

    def _transcribe(self) -> str | None:
        """Transcribe the current audio buffer. Returns text or None."""
        if self._model is None or len(self.audio_buffer) < 1600:  # < 0.1s
            return None
        try:
            # Convert int16 to float32 in [-1, 1] as required by faster-whisper
            audio_f32 = self.audio_buffer.astype(np.float32) / 32768.0
            segments, _ = self._model.transcribe(
                audio_f32,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            text = "".join(seg.text for seg in segments)
            return text.strip()
        except Exception as e:
            log.error("Transcription error: %s", e)
            return None

    def run(self) -> None:
        """Thread main loop: load model, then poll queue and transcribe."""
        if not self._load_model():
            return
        self._running = True
        while self._running:
            self.msleep(self.POLL_INTERVAL_MS)
            self.accumulate()
            text = self._transcribe()
            if text is not None:
                self.transcription_updated.emit(text)

    def stop(self) -> None:
        """Signal the worker to stop after current iteration."""
        self._running = False
