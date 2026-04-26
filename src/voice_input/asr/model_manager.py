from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from voice_input.config import xdg_cache_dir

log = logging.getLogger(__name__)


# SHA256 在实施时从 HuggingFace 实际下载文件用 sha256sum 计算并填入。
# 占位常量，运行时下载完成后会校验；首次实施需替换为真实值。
PARAFORMER_MODEL_SHA256 = ""
PARAFORMER_TOKENS_SHA256 = ""
SENSE_VOICE_MODEL_SHA256 = ""
SENSE_VOICE_TOKENS_SHA256 = ""
SILERO_VAD_SHA256 = ""


@dataclass(frozen=True)
class ModelMeta:
    family: str
    base_url: str
    files: dict[str, str]
    sha256: dict[str, str]
    language: str
    size_bytes: int


@dataclass
class ModelInfo:
    model_id: str
    family: str
    paths: dict[str, Path]
    language: str
    size_bytes: int


@dataclass
class ModelSummary:
    model_id: str
    family: str
    language: str
    installed: bool
    size_bytes: int


REGISTRY: dict[str, ModelMeta] = {
    "sherpa-onnx-paraformer-zh-2024-03-09": ModelMeta(
        family="paraformer",
        base_url="https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09/resolve/main/",
        files={"model": "model.int8.onnx", "tokens": "tokens.txt"},
        sha256={"model": PARAFORMER_MODEL_SHA256, "tokens": PARAFORMER_TOKENS_SHA256},
        language="zh-en",
        size_bytes=237_000_000,
    ),
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17": ModelMeta(
        family="sense_voice",
        base_url="https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/",
        files={"model": "model.int8.onnx", "tokens": "tokens.txt"},
        sha256={"model": SENSE_VOICE_MODEL_SHA256, "tokens": SENSE_VOICE_TOKENS_SHA256},
        language="zh-en-ja-ko-yue",
        size_bytes=234_000_000,
    ),
}

VAD_META = ModelMeta(
    family="vad",
    base_url="https://huggingface.co/csukuangfj/sherpa-onnx-silero-vad/resolve/main/",
    files={"model": "silero_vad.onnx"},
    sha256={"model": SILERO_VAD_SHA256},
    language="any",
    size_bytes=1_800_000,
)


class ModelManager:
    """Sherpa-onnx 模型下载、校验、缓存管理。"""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or (xdg_cache_dir() / "sherpa-models")

    def _model_dir(self, model_id: str) -> Path:
        return self.base_dir / model_id
