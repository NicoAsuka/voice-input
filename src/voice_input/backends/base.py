from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class RecognitionError(Exception):
    """识别期间的可恢复错误，上层应弹 toast 提示用户。"""

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or "识别失败，请重试"


@dataclass(frozen=True)
class BackendCapabilities:
    supports_streaming: bool = False
    requires_network: bool = False
    supports_vad: bool = False


@dataclass(frozen=True)
class BackendDescriptor:
    backend_id: str
    model_id: str
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)


class Session(ABC):
    """单次识别会话。生命周期：create → push_audio* → finish → final_text"""

    @abstractmethod
    def push_audio(self, pcm_int16: np.ndarray) -> None: ...

    @abstractmethod
    async def finish(self) -> str:
        """终止录音、等待识别完成、返回最终文本。可能抛 RecognitionError。"""

    @abstractmethod
    def cancel(self) -> None: ...


class TranscriptionBackend(ABC):
    """长生命周期，持有模型/连接。"""

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    def describe(self) -> BackendDescriptor: ...

    @abstractmethod
    def create_session(self, language: str) -> Session: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    def is_ready(self) -> bool:
        """同步快速检查，不阻塞。BackendRegistry 用它判断当前是否可用。"""
        return True
