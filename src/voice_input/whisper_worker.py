# src/voice_input/whisper_worker.py
from __future__ import annotations

import logging
import queue

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from voice_input.config import xdg_cache_dir

log = logging.getLogger(__name__)

# Max audio to transcribe at once (seconds). Keeps transcription fast.
MAX_BUFFER_SECONDS = 30
SAMPLE_RATE = 16000
MAX_BUFFER_SAMPLES = MAX_BUFFER_SECONDS * SAMPLE_RATE
MIN_RMS = 32.0
MIN_PEAK = 256


class WhisperWorker(QThread):
    """Runs faster-whisper transcription in a background thread.

    Polls the audio queue, accumulates into a buffer (capped at 30s),
    and runs transcription. Only active while recording.
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
        self._active = False

    def set_active(self, active: bool) -> None:
        self._active = active

    def reset(self) -> None:
        self.audio_buffer = np.array([], dtype=np.int16)
        # Drain leftover audio
        self.drain_queue()

    def drain_queue(self) -> np.ndarray:
        """Drain all available chunks from the queue."""
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
        """Append queued audio to the cumulative audio buffer."""
        new_data = self.drain_queue()
        if len(new_data) == 0:
            return
        self.audio_buffer = np.concatenate([self.audio_buffer, new_data])
        if len(self.audio_buffer) > MAX_BUFFER_SAMPLES:
            self.audio_buffer = self.audio_buffer[-MAX_BUFFER_SAMPLES:]

    def _drain_and_accumulate(self) -> None:
        self.accumulate()

    def get_audio_buffer(self) -> np.ndarray:
        """Return a copy of the current audio buffer."""
        self._drain_and_accumulate()
        if len(self.audio_buffer):
            rms = float(np.sqrt(np.mean(self.audio_buffer.astype(np.float64) ** 2)))
            peak = int(np.max(np.abs(self.audio_buffer)))
            log.info(
                "Captured audio buffer: duration=%.2fs rms=%.1f peak=%d",
                len(self.audio_buffer) / SAMPLE_RATE,
                rms,
                peak,
            )
        return self.audio_buffer.copy()

    def _load_model(self) -> bool:
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
            log.info("Whisper model loaded and ready")
            self.model_ready.emit()
            return True
        except Exception as e:
            log.error("Failed to load whisper model: %s", e)
            self.error_occurred.emit(f"Whisper model load failed: {e}")
            return False

    def _transcribe(self) -> str | None:
        if self._model is None or len(self.audio_buffer) < 1600:
            return None
        rms = float(np.sqrt(np.mean(self.audio_buffer.astype(np.float64) ** 2)))
        peak = int(np.max(np.abs(self.audio_buffer))) if len(self.audio_buffer) else 0
        if rms < MIN_RMS or peak < MIN_PEAK:
            return None
        try:
            audio_f32 = self.audio_buffer.astype(np.float32) / 32768.0
            segments, _ = self._model.transcribe(
                audio_f32,
                language=self.language,
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
            return None

    def run(self) -> None:
        if not self._load_model():
            return
        self._running = True
        while self._running:
            self.msleep(self.POLL_INTERVAL_MS)
            if not self._active:
                continue
            self._drain_and_accumulate()
            text = self._transcribe()
            if text is not None:
                log.debug("Transcription: %s", text[:80])
                self.transcription_updated.emit(text)

    def stop(self) -> None:
        self._running = False
