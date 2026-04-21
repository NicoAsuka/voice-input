# src/voice_input/whisper_worker.py
from __future__ import annotations

import asyncio
import logging
import queue

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

MAX_BUFFER_SECONDS = 30
SAMPLE_RATE = 16000
MAX_BUFFER_SAMPLES = MAX_BUFFER_SECONDS * SAMPLE_RATE
MIN_TRANSCRIBE_SAMPLES = 1600


class WhisperWorker(QThread):
    """Runs transcription in a background thread.

    Streaming backends emit incremental transcription while recording is active.
    Non-streaming backends only accumulate audio; transcription happens after
    recording stops.
    """

    transcription_updated = pyqtSignal(str)
    model_ready = pyqtSignal()
    error_occurred = pyqtSignal(str)

    POLL_INTERVAL_MS = 500

    def __init__(
        self,
        whisper_queue: queue.Queue,
        backend,
        language: str = "zh",
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.whisper_queue = whisper_queue
        self._backend = backend
        self.language = language
        self.audio_buffer = np.array([], dtype=np.int16)
        self._running = False
        self._active = False

    def set_active(self, active: bool) -> None:
        self._active = active

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
        """Drain queue and append to the capped cumulative audio buffer."""
        new_data = self.drain_queue()
        if len(new_data) == 0:
            return
        self.audio_buffer = np.concatenate([self.audio_buffer, new_data])
        if len(self.audio_buffer) > MAX_BUFFER_SAMPLES:
            self.audio_buffer = self.audio_buffer[-MAX_BUFFER_SAMPLES:]

    def reset(self) -> None:
        """Clear the audio buffer for a new recording session."""
        self.audio_buffer = np.array([], dtype=np.int16)
        self.drain_queue()

    def get_buffer(self) -> np.ndarray:
        """Return a copy of the current audio buffer."""
        return self.audio_buffer.copy()

    def run(self) -> None:
        """Thread main loop: initialize backend, then poll queue."""
        try:
            asyncio.run(self._backend.initialize())
        except Exception as e:
            log.error("Backend initialization failed: %s", e)
            self.error_occurred.emit(f"Model load failed: {e}")
            return

        self.model_ready.emit()
        self._running = True
        streaming = self._backend.is_streaming()
        while self._running:
            self.msleep(self.POLL_INTERVAL_MS)
            if not self._active:
                continue

            self.accumulate()
            if streaming and len(self.audio_buffer) >= MIN_TRANSCRIBE_SAMPLES:
                try:
                    text = asyncio.run(
                        self._backend.transcribe(self.audio_buffer, self.language)
                    )
                    if text:
                        log.debug("Transcription: %s", text[:80])
                        self.transcription_updated.emit(text)
                except Exception as e:
                    log.error("Transcription error: %s", e)

    def stop(self) -> None:
        """Signal the worker to stop after current iteration."""
        self._running = False
