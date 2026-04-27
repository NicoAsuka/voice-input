from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

try:
    import sherpa_onnx  # type: ignore
except ImportError:
    sherpa_onnx = None  # type: ignore[assignment]

from voice_input.asr.model_manager import ModelManager
from voice_input.asr.vad import VadTrimmer
from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    RecognitionError,
    Session,
    TranscriptionBackend,
)

log = logging.getLogger(__name__)


def _build_recognizer(model_info: Any, num_threads: int, provider: str) -> Any:
    """根据 family 选择正确的 sherpa-onnx 工厂方法。"""
    if sherpa_onnx is None:
        raise RuntimeError("sherpa_onnx is not installed")

    family = model_info.family
    if family == "paraformer":
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(model_info.paths["model"]),
            tokens=str(model_info.paths["tokens"]),
            num_threads=num_threads,
            provider=provider,
            sample_rate=16000,
            feature_dim=80,
        )
    if family == "sense_voice":
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_info.paths["model"]),
            tokens=str(model_info.paths["tokens"]),
            num_threads=num_threads,
            provider=provider,
            use_itn=True,
        )
    raise ValueError(f"Unsupported model family: {family}")


class SherpaSession(Session):
    """单次识别会话（offline / buffered 模式）。"""

    def __init__(self, recognizer: Any, vad: VadTrimmer | None, language: str) -> None:
        self._recognizer = recognizer
        self._vad = vad
        self._language = language
        self._buffer: list[np.ndarray] = []
        self._cancelled = False

    def push_audio(self, pcm_int16: np.ndarray) -> None:
        if self._cancelled:
            return
        self._buffer.append(pcm_int16)

    def cancel(self) -> None:
        self._cancelled = True
        self._buffer.clear()

    async def finish(self) -> str:
        if self._cancelled or not self._buffer:
            return ""

        pcm = np.concatenate(self._buffer)
        audio_f32 = pcm.astype(np.float32) / 32768.0

        if self._vad is not None and self._vad.available():
            audio_f32 = self._vad.trim(audio_f32, sample_rate=16000)
            if len(audio_f32) == 0:
                return ""

        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, self._recognize_sync, audio_f32)
        except Exception as e:
            log.exception("sherpa inference failed")
            raise RecognitionError(
                str(e), user_message="识别引擎错误，请检查模型"
            ) from e
        return text.strip()

    def _recognize_sync(self, audio_f32: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, audio_f32)
        self._recognizer.decode_stream(stream)
        return stream.result.text


class SherpaBackend(TranscriptionBackend):
    """sherpa-onnx 本地识别 backend。"""

    def __init__(self, config: dict) -> None:
        sherpa_cfg = config.get("stt", {}).get("sherpa", {})
        self._model_id = sherpa_cfg.get(
            "model_id", "sherpa-onnx-paraformer-zh-2024-03-09"
        )
        self._vad_enabled = sherpa_cfg.get("vad_enabled", True)
        self._num_threads = sherpa_cfg.get("num_threads", 2)
        self._provider = sherpa_cfg.get("provider", "cpu")
        self._recognizer: Any = None
        self._vad: VadTrimmer | None = None
        self._model_info: Any = None

    def is_ready(self) -> bool:
        return self._recognizer is not None

    def describe(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id="sherpa",
            model_id=self._model_id,
            capabilities=BackendCapabilities(
                supports_streaming=False,
                requires_network=False,
                supports_vad=self._vad_enabled,
            ),
        )

    async def initialize(self) -> None:
        manager = ModelManager()
        self._model_info = await manager.ensure_model(self._model_id)

        loop = asyncio.get_running_loop()
        self._recognizer = await loop.run_in_executor(
            None,
            _build_recognizer,
            self._model_info,
            self._num_threads,
            self._provider,
        )

        if self._vad_enabled:
            vad = VadTrimmer()
            vad_path = await manager.ensure_vad_model()
            await loop.run_in_executor(None, vad.load, vad_path)
            self._vad = vad

        log.info(
            "SherpaBackend initialized with model=%s vad=%s",
            self._model_id,
            self._vad_enabled,
        )

    def create_session(self, language: str) -> Session:
        if not self.is_ready():
            raise RuntimeError("SherpaBackend not initialized")
        return SherpaSession(
            recognizer=self._recognizer,
            vad=self._vad,
            language=language,
        )

    async def shutdown(self) -> None:
        self._recognizer = None
        self._vad = None
        log.info("SherpaBackend shutdown")
