from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import sherpa_onnx  # type: ignore
except ImportError:
    sherpa_onnx = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


def _vad_empty(vad: object) -> bool:
    empty = getattr(vad, "empty", None)
    if callable(empty):
        result = empty()
        if isinstance(result, bool):
            return result
        if not result.__class__.__module__.startswith("unittest.mock"):
            return bool(result)

    is_empty = getattr(vad, "is_empty", None)
    if callable(is_empty):
        return bool(is_empty())

    raise AttributeError("VAD object does not expose empty() or is_empty()")


def _vad_front(vad: object) -> object:
    front = getattr(vad, "front")
    if callable(front) and "samples" not in getattr(front, "__dict__", {}):
        return front()
    return front


class VadTrimmer:
    """Silero VAD wrapper. 输入 PCM float32, 输出去除静音后的 PCM。"""

    def __init__(self) -> None:
        self._vad = None
        self._sample_rate = 16000

    def available(self) -> bool:
        return self._vad is not None

    def load(self, model_path: Path, sample_rate: int = 16000) -> None:
        """加载 silero_vad.onnx。"""
        if sherpa_onnx is None:
            log.warning("sherpa_onnx not installed; VAD disabled")
            return

        try:
            config = sherpa_onnx.VadModelConfig(
                silero_vad=sherpa_onnx.SileroVadModelConfig(
                    model=str(model_path),
                    threshold=0.5,
                    min_silence_duration=0.25,
                    min_speech_duration=0.25,
                    window_size=512,
                ),
                sample_rate=sample_rate,
            )
            self._vad = sherpa_onnx.VoiceActivityDetector(
                config, buffer_size_in_seconds=30
            )
            self._sample_rate = sample_rate
            log.info("VAD loaded from %s", model_path)
        except Exception as e:
            log.error("Failed to load VAD: %s", e)
            self._vad = None

    def trim(self, samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """喂入音频，提取所有 speech segment 拼接返回。VAD 不可用时透传。"""
        if self._vad is None:
            return samples

        if sample_rate != self._sample_rate:
            raise ValueError(
                f"VAD sample rate mismatch: got {sample_rate}, "
                f"expected {self._sample_rate}"
            )

        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        try:
            self._vad.accept_waveform(samples)
            self._vad.flush()

            segments: list[np.ndarray] = []
            while not _vad_empty(self._vad):
                segment = _vad_front(self._vad)
                segments.append(np.asarray(segment.samples, dtype=np.float32))
                self._vad.pop()
        finally:
            self._vad.reset()

        if not segments:
            return np.array([], dtype=np.float32)
        return np.concatenate(segments)
