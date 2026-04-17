# src/voice_input/audio.py
from __future__ import annotations

import logging
import queue

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


def compute_rms(samples: np.ndarray) -> float:
    """Compute RMS level normalized to [0.0, 1.0] for int16 audio."""
    float_samples = samples.astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(float_samples ** 2)))


class AudioRecorder:
    """Captures 16kHz mono int16 audio via sounddevice.

    Pushes raw frames to whisper_queue (for ASR) and viz_queue (for waveform).
    """

    def __init__(
        self,
        whisper_queue: queue.Queue,
        viz_queue: queue.Queue,
        device: str = "default",
        sample_rate: int = 16000,
    ) -> None:
        self.whisper_queue = whisper_queue
        self.viz_queue = viz_queue
        self.device = device if device != "default" else None
        self.sample_rate = sample_rate
        self._stream: sd.InputStream | None = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self) -> None:
        if self.is_recording:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
            blocksize=int(self.sample_rate * 0.032),  # ~32ms blocks
            callback=self._callback,
        )
        self._stream.start()
        log.info("Recording started (device=%s, rate=%d)", self.device, self.sample_rate)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Recording stopped")

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            log.warning("sounddevice status: %s", status)
        data = indata[:, 0].copy()  # mono channel, copy to decouple from buffer
        try:
            self.whisper_queue.put_nowait(data)
        except queue.Full:
            pass
        try:
            self.viz_queue.put_nowait(data)
        except queue.Full:
            pass
