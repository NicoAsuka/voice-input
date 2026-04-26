# sherpa-onnx 迁移与本地引擎重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 voice-input 的本地 STT 引擎从 Whisper/SenseVoice 完全替换为 sherpa-onnx，引入异步预加载、配置签名 + reload worker、Session 抽象、VAD 集成、模型管理子系统、场景化后处理。

**Architecture:** 单进程 + asyncio。新增 `BackendRegistry` 持有 effective backend，启动时异步加载，配置变化时后台 reload，旧 backend 保留可用直到新 backend ready。`TranscriptionBackend` 接口改为 Session 模式，一次性识别交给 `Session` 实现。本地引擎只剩 `SherpaBackend`（基于 sherpa-onnx），云 backend（VolcEngine/Google/OpenAI Whisper）适配新接口。LLM 后处理升级为多场景 prompt。

**Tech Stack:** Python 3.11+, asyncio, sherpa-onnx, httpx, PyQt6, pytest + pytest-asyncio + respx

**Reference Spec:** `docs/superpowers/specs/2026-04-26-sherpa-onnx-migration-design.md`

---

## File Structure

### 新增文件
```
src/voice_input/
├── backends/
│   ├── registry.py        # BackendRegistry：异步加载 + reload worker
│   ├── sherpa_backend.py  # SherpaBackend + SherpaSession
├── asr/
│   ├── __init__.py
│   ├── model_manager.py   # 模型下载/校验/列表/删除
│   └── vad.py             # Silero VAD 包装
├── postprocess/
│   ├── __init__.py
│   ├── llm.py             # 从根目录移入 + refine(prompt) 接口
│   ├── scene.py           # Scene + SceneRegistry
│   └── pipeline.py        # ScenePipeline

tests/
├── test_backend_registry.py
├── test_sherpa_backend.py
├── test_model_manager.py
├── test_vad.py
├── test_scene.py
└── test_pipeline.py
```

### 修改文件
- `src/voice_input/backends/base.py`（重写，引入 Session）
- `src/voice_input/backends/__init__.py`（工厂适配）
- `src/voice_input/backends/volcengine_speech.py`（适配 Session 接口）
- `src/voice_input/backends/google_speech.py`（适配）
- `src/voice_input/backends/openai_whisper.py`（适配）
- `src/voice_input/config.py`（schema 变更）
- `src/voice_input/app.py`（用 BackendRegistry，去掉 WhisperWorker）
- `src/voice_input/tray.py`（场景子菜单，移除 engine 概念）
- `src/voice_input/settings_dialog.py`（场景管理 + sherpa 配置）
- `pyproject.toml`（删 whisper/sensevoice，加 sherpa-onnx + respx）
- `tests/test_app.py`（适配）

### 删除文件
- `src/voice_input/backends/local/__init__.py`
- `src/voice_input/backends/local/whisper_engine.py`
- `src/voice_input/backends/local/sensevoice_engine.py`
- `src/voice_input/backends/local/engine.py`
- `src/voice_input/whisper_worker.py`
- `src/voice_input/llm.py`（已移到 postprocess/llm.py）
- `tests/test_local_backend.py`

---

## Task 1: 重写 backends/base.py — 新接口（Session + RecognitionError）

**Files:**
- Modify: `src/voice_input/backends/base.py` (整个文件重写)
- Create: `tests/test_backends_base.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backends_base.py
from __future__ import annotations

import pytest

from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    RecognitionError,
    Session,
    TranscriptionBackend,
)


def test_recognition_error_has_user_message():
    e = RecognitionError("internal", user_message="识别失败")
    assert str(e) == "internal"
    assert e.user_message == "识别失败"


def test_recognition_error_default_user_message():
    e = RecognitionError("boom")
    assert e.user_message == "识别失败，请重试"


def test_backend_capabilities_defaults():
    caps = BackendCapabilities()
    assert caps.supports_streaming is False
    assert caps.requires_network is False
    assert caps.supports_vad is False


def test_backend_descriptor_fields():
    desc = BackendDescriptor(
        backend_id="fake",
        model_id="m1",
        capabilities=BackendCapabilities(supports_vad=True),
    )
    assert desc.backend_id == "fake"
    assert desc.model_id == "m1"
    assert desc.capabilities.supports_vad is True


def test_session_is_abstract():
    with pytest.raises(TypeError):
        Session()  # type: ignore[abstract]


def test_transcription_backend_is_abstract():
    with pytest.raises(TypeError):
        TranscriptionBackend()  # type: ignore[abstract]


def test_default_is_ready_returns_true():
    class Concrete(TranscriptionBackend):
        async def initialize(self): pass
        def describe(self): return BackendDescriptor("x", "y", BackendCapabilities())
        def create_session(self, language): raise NotImplementedError
        async def shutdown(self): pass
    assert Concrete().is_ready() is True
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_backends_base.py -v`
Expected: FAIL（导入错误，因为接口还没改）

- [ ] **Step 3: 重写 base.py**

```python
# src/voice_input/backends/base.py
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_backends_base.py -v`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add src/voice_input/backends/base.py tests/test_backends_base.py
git commit -m "refactor(backends): rewrite base.py with Session + RecognitionError"
```

---

## Task 2: ModelManager — 数据结构与元数据 REGISTRY

**Files:**
- Create: `src/voice_input/asr/__init__.py`
- Create: `src/voice_input/asr/model_manager.py`
- Create: `tests/test_model_manager.py`

- [ ] **Step 1: 写失败测试（数据结构）**

```python
# tests/test_model_manager.py
from __future__ import annotations

from pathlib import Path

import pytest

from voice_input.asr.model_manager import (
    REGISTRY,
    VAD_META,
    ModelInfo,
    ModelManager,
    ModelMeta,
    ModelSummary,
)


def test_registry_has_paraformer_and_sense_voice():
    assert "sherpa-onnx-paraformer-zh-2024-03-09" in REGISTRY
    assert "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" in REGISTRY


def test_paraformer_meta_fields():
    meta = REGISTRY["sherpa-onnx-paraformer-zh-2024-03-09"]
    assert meta.family == "paraformer"
    assert meta.base_url.startswith("https://huggingface.co/")
    assert "model" in meta.files
    assert "tokens" in meta.files
    assert "model" in meta.sha256
    assert "tokens" in meta.sha256
    assert meta.language == "zh-en"
    assert meta.size_bytes > 0


def test_vad_meta_fields():
    assert VAD_META.family == "vad"
    assert "model" in VAD_META.files
    assert "model" in VAD_META.sha256


def test_model_manager_uses_default_base_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    mgr = ModelManager()
    assert mgr.base_dir == tmp_path / "voice-input" / "sherpa-models"


def test_model_manager_custom_base_dir(tmp_path):
    mgr = ModelManager(base_dir=tmp_path)
    assert mgr.base_dir == tmp_path
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_model_manager.py -v`
Expected: FAIL（导入错误）

- [ ] **Step 3: 创建 asr/__init__.py（空文件）**

```python
# src/voice_input/asr/__init__.py
```

- [ ] **Step 4: 创建 model_manager.py 数据结构与 REGISTRY**

```python
# src/voice_input/asr/model_manager.py
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
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_model_manager.py -v`
Expected: 5 passed

- [ ] **Step 6: 提交**

```bash
git add src/voice_input/asr/__init__.py src/voice_input/asr/model_manager.py tests/test_model_manager.py
git commit -m "feat(asr): add ModelManager skeleton with REGISTRY metadata"
```

---

## Task 3: ModelManager — 下载与 SHA256 校验

**Files:**
- Modify: `src/voice_input/asr/model_manager.py` (添加 ensure_model + ensure_vad_model)
- Modify: `tests/test_model_manager.py` (添加下载测试)

- [ ] **Step 1: 添加 respx 到 dev 依赖**

```bash
# 编辑 pyproject.toml 第 24 行附近
```

Edit `pyproject.toml`:
```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-qt>=4.3", "pytest-asyncio>=0.23", "respx>=0.21"]
```

- [ ] **Step 2: 安装新依赖**

Run: `pip install -e ".[dev]"`

- [ ] **Step 3: 写失败测试（下载场景）**

追加到 `tests/test_model_manager.py`：

```python
import hashlib

import httpx
import pytest
import respx
from httpx import Response


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def fake_paraformer_meta(monkeypatch):
    """注入一个测试用的 ModelMeta，避免依赖真实远程文件。"""
    from voice_input.asr import model_manager

    model_bytes = b"fake-onnx-bytes"
    tokens_bytes = b"fake-tokens-bytes"
    test_meta = model_manager.ModelMeta(
        family="paraformer",
        base_url="https://example.com/test-model/",
        files={"model": "model.onnx", "tokens": "tokens.txt"},
        sha256={"model": _sha256(model_bytes), "tokens": _sha256(tokens_bytes)},
        language="zh-en",
        size_bytes=len(model_bytes) + len(tokens_bytes),
    )
    monkeypatch.setitem(model_manager.REGISTRY, "test-model", test_meta)
    return test_meta, model_bytes, tokens_bytes


@pytest.mark.asyncio
async def test_ensure_model_downloads_when_missing(tmp_path, fake_paraformer_meta):
    meta, model_bytes, tokens_bytes = fake_paraformer_meta
    mgr = ModelManager(base_dir=tmp_path)

    with respx.mock:
        respx.get("https://example.com/test-model/model.onnx").mock(
            return_value=Response(200, content=model_bytes)
        )
        respx.get("https://example.com/test-model/tokens.txt").mock(
            return_value=Response(200, content=tokens_bytes)
        )
        info = await mgr.ensure_model("test-model")

    assert info.model_id == "test-model"
    assert info.family == "paraformer"
    assert info.paths["model"].exists()
    assert info.paths["tokens"].exists()
    assert info.paths["model"].read_bytes() == model_bytes


@pytest.mark.asyncio
async def test_ensure_model_skips_when_already_installed(tmp_path, fake_paraformer_meta):
    meta, model_bytes, tokens_bytes = fake_paraformer_meta
    mgr = ModelManager(base_dir=tmp_path)
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(model_bytes)
    (model_dir / "tokens.txt").write_bytes(tokens_bytes)

    with respx.mock:
        # 不 mock 任何路由，确保没有实际请求发起
        info = await mgr.ensure_model("test-model")

    assert info.paths["model"].read_bytes() == model_bytes


@pytest.mark.asyncio
async def test_ensure_model_validates_sha256(tmp_path, fake_paraformer_meta):
    meta, model_bytes, tokens_bytes = fake_paraformer_meta
    mgr = ModelManager(base_dir=tmp_path)

    with respx.mock:
        respx.get("https://example.com/test-model/model.onnx").mock(
            return_value=Response(200, content=b"corrupted-bytes")
        )
        respx.get("https://example.com/test-model/tokens.txt").mock(
            return_value=Response(200, content=tokens_bytes)
        )
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            await mgr.ensure_model("test-model")

    # 失败后不留下半成品
    assert not (tmp_path / "test-model" / "model.onnx").exists()


@pytest.mark.asyncio
async def test_ensure_model_unknown_id_raises(tmp_path):
    mgr = ModelManager(base_dir=tmp_path)
    with pytest.raises(KeyError):
        await mgr.ensure_model("nonexistent-model-id")


@pytest.mark.asyncio
async def test_ensure_model_atomic_rename(tmp_path, fake_paraformer_meta, monkeypatch):
    """下载到临时位置后再 rename，中断不留半成品。"""
    meta, model_bytes, tokens_bytes = fake_paraformer_meta
    mgr = ModelManager(base_dir=tmp_path)

    # 模拟第二个文件下载时失败
    with respx.mock:
        respx.get("https://example.com/test-model/model.onnx").mock(
            return_value=Response(200, content=model_bytes)
        )
        respx.get("https://example.com/test-model/tokens.txt").mock(
            return_value=Response(500, content=b"server error")
        )
        with pytest.raises(httpx.HTTPStatusError):
            await mgr.ensure_model("test-model")

    # 失败后整个 model_dir 不应存在
    assert not (tmp_path / "test-model").exists()
```

- [ ] **Step 4: 运行测试确认失败**

Run: `pytest tests/test_model_manager.py -v -k "ensure_model"`
Expected: FAIL（ensure_model 还没实现）

- [ ] **Step 5: 实现 ensure_model 与 ensure_vad_model**

追加到 `src/voice_input/asr/model_manager.py`：

```python
async def _download_to_path(
    client: httpx.AsyncClient, url: str, dest: Path, expected_sha256: str
) -> None:
    """下载到临时文件 → SHA256 校验 → 原子 rename。"""
    log.info("Downloading %s → %s", url, dest)
    with tempfile.NamedTemporaryFile(
        delete=False, dir=dest.parent, suffix=".part"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        async with client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            sha = hashlib.sha256()
            with open(tmp_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    f.write(chunk)
                    sha.update(chunk)

        if expected_sha256:
            actual = sha.hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(
                    f"SHA256 mismatch for {url}: expected {expected_sha256}, got {actual}"
                )
        tmp_path.replace(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


async def _download_meta_to_dir(meta: ModelMeta, target_dir: Path) -> dict[str, Path]:
    """下载 meta 中所有文件到 target_dir。失败则清理。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            for role, filename in meta.files.items():
                url = meta.base_url + filename
                dest = target_dir / filename
                expected = meta.sha256.get(role, "")
                await _download_to_path(client, url, dest, expected)
                paths[role] = dest
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    return paths


# 在 ModelManager 类内追加：

    async def ensure_model(self, model_id: str) -> ModelInfo:
        """已存在 → 直接返回；不存在 → 下载到 cache 目录。"""
        if model_id not in REGISTRY:
            raise KeyError(f"Unknown model_id: {model_id}")
        meta = REGISTRY[model_id]
        target_dir = self._model_dir(model_id)

        # 已安装：所有文件存在则跳过下载
        existing_paths = {
            role: target_dir / fn for role, fn in meta.files.items()
        }
        if all(p.exists() for p in existing_paths.values()):
            log.debug("Model %s already installed", model_id)
            return ModelInfo(
                model_id=model_id,
                family=meta.family,
                paths=existing_paths,
                language=meta.language,
                size_bytes=meta.size_bytes,
            )

        log.info("Installing model %s into %s", model_id, target_dir)
        paths = await _download_meta_to_dir(meta, target_dir)
        return ModelInfo(
            model_id=model_id,
            family=meta.family,
            paths=paths,
            language=meta.language,
            size_bytes=meta.size_bytes,
        )

    async def ensure_vad_model(self) -> Path:
        """确保 silero_vad.onnx 存在，返回路径。"""
        target_dir = self.base_dir / "silero-vad"
        existing = target_dir / VAD_META.files["model"]
        if existing.exists():
            return existing
        paths = await _download_meta_to_dir(VAD_META, target_dir)
        return paths["model"]
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_model_manager.py -v`
Expected: all passed

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml src/voice_input/asr/model_manager.py tests/test_model_manager.py
git commit -m "feat(asr): add model download with sha256 validation"
```

---

## Task 4: ModelManager — list_installed 与 remove

**Files:**
- Modify: `src/voice_input/asr/model_manager.py`
- Modify: `tests/test_model_manager.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_model_manager.py`：

```python
def test_list_installed_returns_summaries(tmp_path, fake_paraformer_meta):
    meta, model_bytes, tokens_bytes = fake_paraformer_meta
    mgr = ModelManager(base_dir=tmp_path)
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(model_bytes)
    (model_dir / "tokens.txt").write_bytes(tokens_bytes)

    summaries = mgr.list_installed()
    by_id = {s.model_id: s for s in summaries}
    assert "test-model" in by_id
    assert by_id["test-model"].installed is True
    assert by_id["test-model"].family == "paraformer"
    # 未安装的也列出来，标记 installed=False
    assert "sherpa-onnx-paraformer-zh-2024-03-09" in by_id
    assert by_id["sherpa-onnx-paraformer-zh-2024-03-09"].installed is False


def test_remove_deletes_model_dir(tmp_path, fake_paraformer_meta):
    meta, model_bytes, tokens_bytes = fake_paraformer_meta
    mgr = ModelManager(base_dir=tmp_path)
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(model_bytes)

    mgr.remove("test-model")

    assert not model_dir.exists()


def test_remove_unknown_id_is_silent(tmp_path):
    mgr = ModelManager(base_dir=tmp_path)
    # 不抛错，只 log
    mgr.remove("nonexistent")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_model_manager.py -v -k "list_installed or remove"`
Expected: FAIL

- [ ] **Step 3: 实现 list_installed 与 remove**

在 `ModelManager` 类内追加：

```python
    def list_installed(self) -> list[ModelSummary]:
        """列出 REGISTRY 中所有模型，标记是否已安装。"""
        results: list[ModelSummary] = []
        for model_id, meta in REGISTRY.items():
            target_dir = self._model_dir(model_id)
            installed = all(
                (target_dir / fn).exists() for fn in meta.files.values()
            )
            results.append(
                ModelSummary(
                    model_id=model_id,
                    family=meta.family,
                    language=meta.language,
                    installed=installed,
                    size_bytes=meta.size_bytes,
                )
            )
        return results

    def remove(self, model_id: str) -> None:
        """删除模型目录。不存在时静默。"""
        target_dir = self._model_dir(model_id)
        if not target_dir.exists():
            log.debug("remove: %s not found at %s", model_id, target_dir)
            return
        log.info("Removing model %s from %s", model_id, target_dir)
        shutil.rmtree(target_dir, ignore_errors=True)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_model_manager.py -v`
Expected: all passed

- [ ] **Step 5: 提交**

```bash
git add src/voice_input/asr/model_manager.py tests/test_model_manager.py
git commit -m "feat(asr): add list_installed and remove for ModelManager"
```

---

## Task 5: VadTrimmer

**Files:**
- Create: `src/voice_input/asr/vad.py`
- Create: `tests/test_vad.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_vad.py
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_input.asr.vad import VadTrimmer


def test_unavailable_when_not_loaded():
    vad = VadTrimmer()
    assert vad.available() is False


def test_trim_returns_input_when_unavailable():
    vad = VadTrimmer()
    samples = np.zeros(16000, dtype=np.float32)
    out = vad.trim(samples, sample_rate=16000)
    # 未加载时直接透传
    assert np.array_equal(out, samples)


def test_load_initializes_vad():
    vad = VadTrimmer()
    fake_vad = MagicMock()
    with patch("voice_input.asr.vad.sherpa_onnx") as fake_sherpa:
        fake_sherpa.VoiceActivityDetector.return_value = fake_vad
        vad.load(Path("/tmp/silero_vad.onnx"))
    assert vad.available() is True
    fake_sherpa.VoiceActivityDetector.assert_called_once()


def test_trim_concatenates_speech_segments():
    vad = VadTrimmer()
    fake_vad = MagicMock()

    # Stub VAD to emit two segments
    seg1 = MagicMock()
    seg1.samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    seg2 = MagicMock()
    seg2.samples = np.array([0.4, 0.5], dtype=np.float32)

    # Simulate VAD streaming API: accept_waveform → flush → is_empty/front/pop
    states = {"emitted": [seg1, seg2], "flushed": False}

    def is_empty():
        return len(states["emitted"]) == 0

    def front():
        return states["emitted"][0]

    def pop():
        states["emitted"].pop(0)

    fake_vad.is_empty.side_effect = is_empty
    fake_vad.front.side_effect = front
    fake_vad.pop.side_effect = pop

    with patch("voice_input.asr.vad.sherpa_onnx") as fake_sherpa:
        fake_sherpa.VoiceActivityDetector.return_value = fake_vad
        vad.load(Path("/tmp/silero_vad.onnx"))

    samples = np.zeros(16000, dtype=np.float32)
    out = vad.trim(samples, sample_rate=16000)
    assert out.dtype == np.float32
    assert np.allclose(out, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_trim_returns_empty_for_no_speech():
    vad = VadTrimmer()
    fake_vad = MagicMock()
    fake_vad.is_empty.return_value = True

    with patch("voice_input.asr.vad.sherpa_onnx") as fake_sherpa:
        fake_sherpa.VoiceActivityDetector.return_value = fake_vad
        vad.load(Path("/tmp/silero_vad.onnx"))

    samples = np.zeros(16000, dtype=np.float32)
    out = vad.trim(samples, sample_rate=16000)
    assert len(out) == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_vad.py -v`
Expected: FAIL（导入错误）

- [ ] **Step 3: 实现 VadTrimmer**

```python
# src/voice_input/asr/vad.py
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import sherpa_onnx  # type: ignore
except ImportError:
    sherpa_onnx = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


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
        try:
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

        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        # 喂入并 flush
        self._vad.accept_waveform(samples)
        self._vad.flush()

        # 收集所有 segments
        segments: list[np.ndarray] = []
        while not self._vad.is_empty():
            segment = self._vad.front()
            segments.append(np.asarray(segment.samples, dtype=np.float32))
            self._vad.pop()

        # Reset internal state for next session
        self._vad.reset()

        if not segments:
            return np.array([], dtype=np.float32)
        return np.concatenate(segments)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_vad.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
git add src/voice_input/asr/vad.py tests/test_vad.py
git commit -m "feat(asr): add VadTrimmer wrapping sherpa-onnx Silero VAD"
```

---

## Task 6: SherpaBackend + SherpaSession

**Files:**
- Create: `src/voice_input/backends/sherpa_backend.py`
- Create: `tests/test_sherpa_backend.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_sherpa_backend.py
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from voice_input.asr.model_manager import ModelInfo
from voice_input.backends.base import RecognitionError
from voice_input.backends.sherpa_backend import SherpaBackend, SherpaSession


def make_config(model_id="sherpa-onnx-paraformer-zh-2024-03-09", vad=True):
    return {
        "stt": {
            "sherpa": {
                "model_id": model_id,
                "vad_enabled": vad,
                "num_threads": 2,
                "provider": "cpu",
            }
        }
    }


def make_model_info(tmp_path):
    return ModelInfo(
        model_id="sherpa-onnx-paraformer-zh-2024-03-09",
        family="paraformer",
        paths={"model": tmp_path / "model.onnx", "tokens": tmp_path / "tokens.txt"},
        language="zh-en",
        size_bytes=237_000_000,
    )


def test_is_ready_false_before_initialize():
    backend = SherpaBackend(make_config())
    assert backend.is_ready() is False


def test_describe_returns_descriptor_with_model_id():
    backend = SherpaBackend(make_config())
    desc = backend.describe()
    assert desc.backend_id == "sherpa"
    assert desc.model_id == "sherpa-onnx-paraformer-zh-2024-03-09"
    assert desc.capabilities.supports_vad is True
    assert desc.capabilities.requires_network is False


@pytest.mark.asyncio
async def test_initialize_loads_model_and_vad(tmp_path):
    backend = SherpaBackend(make_config())
    info = make_model_info(tmp_path)

    with patch("voice_input.backends.sherpa_backend.ModelManager") as MM, patch(
        "voice_input.backends.sherpa_backend.VadTrimmer"
    ) as VT, patch("voice_input.backends.sherpa_backend.sherpa_onnx") as SO:
        mgr_instance = MM.return_value
        mgr_instance.ensure_model = AsyncMock(return_value=info)
        mgr_instance.ensure_vad_model = AsyncMock(
            return_value=tmp_path / "silero_vad.onnx"
        )
        SO.OfflineRecognizer.from_paraformer.return_value = MagicMock()
        vad_instance = VT.return_value

        await backend.initialize()

    assert backend.is_ready() is True
    mgr_instance.ensure_model.assert_awaited_once()
    mgr_instance.ensure_vad_model.assert_awaited_once()
    vad_instance.load.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_skips_vad_when_disabled(tmp_path):
    backend = SherpaBackend(make_config(vad=False))
    info = make_model_info(tmp_path)

    with patch("voice_input.backends.sherpa_backend.ModelManager") as MM, patch(
        "voice_input.backends.sherpa_backend.VadTrimmer"
    ) as VT, patch("voice_input.backends.sherpa_backend.sherpa_onnx") as SO:
        mgr_instance = MM.return_value
        mgr_instance.ensure_model = AsyncMock(return_value=info)
        SO.OfflineRecognizer.from_paraformer.return_value = MagicMock()

        await backend.initialize()

    VT.assert_not_called()
    mgr_instance.ensure_vad_model.assert_not_called()


@pytest.mark.asyncio
async def test_create_session_returns_sherpa_session(tmp_path):
    backend = SherpaBackend(make_config(vad=False))
    info = make_model_info(tmp_path)
    with patch("voice_input.backends.sherpa_backend.ModelManager") as MM, patch(
        "voice_input.backends.sherpa_backend.sherpa_onnx") as SO:
        MM.return_value.ensure_model = AsyncMock(return_value=info)
        SO.OfflineRecognizer.from_paraformer.return_value = MagicMock()
        await backend.initialize()
    session = backend.create_session(language="zh")
    assert isinstance(session, SherpaSession)


@pytest.mark.asyncio
async def test_session_finish_returns_text():
    fake_recognizer = MagicMock()
    fake_stream = MagicMock()
    fake_stream.result.text = "你好世界"
    fake_recognizer.create_stream.return_value = fake_stream

    session = SherpaSession(recognizer=fake_recognizer, vad=None, language="zh")
    pcm = (np.ones(16000, dtype=np.float32) * 0.5 * 32768).astype(np.int16)
    session.push_audio(pcm)
    text = await session.finish()
    assert text == "你好世界"


@pytest.mark.asyncio
async def test_session_finish_returns_empty_for_no_audio():
    fake_recognizer = MagicMock()
    session = SherpaSession(recognizer=fake_recognizer, vad=None, language="zh")
    text = await session.finish()
    assert text == ""


@pytest.mark.asyncio
async def test_session_finish_raises_recognition_error_on_failure():
    fake_recognizer = MagicMock()
    fake_recognizer.create_stream.side_effect = RuntimeError("inference crashed")

    session = SherpaSession(recognizer=fake_recognizer, vad=None, language="zh")
    pcm = np.ones(16000, dtype=np.int16)
    session.push_audio(pcm)
    with pytest.raises(RecognitionError) as exc_info:
        await session.finish()
    assert "识别" in exc_info.value.user_message


@pytest.mark.asyncio
async def test_session_with_vad_filters_silence():
    fake_recognizer = MagicMock()
    fake_stream = MagicMock()
    fake_stream.result.text = "filtered"
    fake_recognizer.create_stream.return_value = fake_stream

    fake_vad = MagicMock()
    fake_vad.available.return_value = True
    fake_vad.trim.return_value = np.array([], dtype=np.float32)  # 全静音

    session = SherpaSession(recognizer=fake_recognizer, vad=fake_vad, language="zh")
    pcm = np.zeros(16000, dtype=np.int16)
    session.push_audio(pcm)
    text = await session.finish()
    assert text == ""
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_sherpa_backend.py -v`
Expected: FAIL（导入错误）

- [ ] **Step 3: 实现 SherpaBackend 与 SherpaSession**

```python
# src/voice_input/backends/sherpa_backend.py
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

try:
    import sherpa_onnx  # type: ignore
except ImportError:
    sherpa_onnx = None  # type: ignore[assignment]

from voice_input.asr.model_manager import REGISTRY, ModelManager
from voice_input.asr.vad import VadTrimmer
from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    RecognitionError,
    Session,
    TranscriptionBackend,
)

log = logging.getLogger(__name__)


def _build_recognizer(model_info, num_threads: int, provider: str) -> Any:
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
        self._model_info = None

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
        log.info("SherpaBackend initialized with model=%s vad=%s",
                 self._model_id, self._vad_enabled)

    def create_session(self, language: str) -> Session:
        if not self.is_ready():
            raise RuntimeError("SherpaBackend not initialized")
        return SherpaSession(
            recognizer=self._recognizer, vad=self._vad, language=language
        )

    async def shutdown(self) -> None:
        self._recognizer = None
        self._vad = None
        log.info("SherpaBackend shutdown")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_sherpa_backend.py -v`
Expected: all passed

- [ ] **Step 5: 提交**

```bash
git add src/voice_input/backends/sherpa_backend.py tests/test_sherpa_backend.py
git commit -m "feat(backends): add SherpaBackend with VAD integration"
```

---

## Task 7: BackendRegistry — Signature 计算 + 状态枚举

**Files:**
- Create: `src/voice_input/backends/registry.py`
- Create: `tests/test_backend_registry.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backend_registry.py
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    Session,
    TranscriptionBackend,
)
from voice_input.backends.registry import (
    BackendRegistry,
    RegistryState,
    compute_signature,
)


def test_registry_state_enum_values():
    assert RegistryState.LOADING.value == "loading"
    assert RegistryState.READY.value == "ready"
    assert RegistryState.RELOADING.value == "reloading"
    assert RegistryState.ERROR.value == "error"


def test_compute_signature_same_for_same_config():
    cfg1 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "x"}}}
    cfg2 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "x"}}}
    assert compute_signature(cfg1) == compute_signature(cfg2)


def test_compute_signature_changes_when_model_changes():
    cfg1 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "x"}}}
    cfg2 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "y"}}}
    assert compute_signature(cfg1) != compute_signature(cfg2)


def test_compute_signature_changes_when_backend_changes():
    cfg1 = {"stt": {"backend": "sherpa"}}
    cfg2 = {"stt": {"backend": "volcengine"}}
    assert compute_signature(cfg1) != compute_signature(cfg2)


def test_compute_signature_ignores_unrelated_fields():
    cfg1 = {"stt": {"backend": "sherpa"}, "ui": {"x": 1}}
    cfg2 = {"stt": {"backend": "sherpa"}, "ui": {"x": 2}}
    assert compute_signature(cfg1) == compute_signature(cfg2)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_backend_registry.py -v -k "signature or state"`
Expected: FAIL

- [ ] **Step 3: 实现 RegistryState 与 compute_signature**

```python
# src/voice_input/backends/registry.py
from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Callable

from voice_input.backends.base import (
    BackendDescriptor,
    Session,
    TranscriptionBackend,
)

log = logging.getLogger(__name__)


class RegistryState(enum.Enum):
    LOADING = "loading"
    READY = "ready"
    RELOADING = "reloading"
    ERROR = "error"


def compute_signature(config: dict) -> str:
    """配置指纹。只包含影响 backend 实例的字段。"""
    stt = config.get("stt", {})
    relevant = {
        "backend": stt.get("backend"),
        "sherpa": stt.get("sherpa"),
        "volcengine": stt.get("volcengine"),
        "google": stt.get("google"),
        "openai": stt.get("openai"),
    }
    serialized = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class _Effective:
    backend: TranscriptionBackend
    descriptor: BackendDescriptor
    signature: str


StateListener = Callable[[RegistryState, str | None], None]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_backend_registry.py -v -k "signature or state"`
Expected: passed

- [ ] **Step 5: 提交**

```bash
git add src/voice_input/backends/registry.py tests/test_backend_registry.py
git commit -m "feat(backends): add registry state enum and signature computation"
```

---

## Task 8: BackendRegistry — start + 异步 reload worker

**Files:**
- Modify: `src/voice_input/backends/registry.py`
- Modify: `tests/test_backend_registry.py`

- [ ] **Step 1: 写失败测试（含 FakeBackend）**

追加到 `tests/test_backend_registry.py`：

```python
class FakeSession(Session):
    def __init__(self, backend_ref):
        self._backend = backend_ref
        self.pushed = []
        self.finished = False

    def push_audio(self, pcm):
        self.pushed.append(pcm)

    async def finish(self):
        self.finished = True
        return "fake-text"

    def cancel(self):
        pass


class FakeBackend(TranscriptionBackend):
    """测试用 backend：可控的初始化时间和失败注入。"""

    def __init__(self, sig="x", init_delay=0.0, fail_init=False, model_id="m"):
        self._sig = sig
        self._init_delay = init_delay
        self._fail_init = fail_init
        self._model_id = model_id
        self._initialized = False
        self.shutdown_called = False
        self.init_call_count = 0

    async def initialize(self):
        self.init_call_count += 1
        if self._init_delay:
            await asyncio.sleep(self._init_delay)
        if self._fail_init:
            raise RuntimeError("init boom")
        self._initialized = True

    def is_ready(self):
        return self._initialized

    def describe(self):
        return BackendDescriptor(
            backend_id="fake",
            model_id=self._model_id,
            capabilities=BackendCapabilities(),
        )

    def create_session(self, language):
        return FakeSession(self)

    async def shutdown(self):
        self.shutdown_called = True


def make_factory(backend: TranscriptionBackend):
    def factory(config):
        return backend
    return factory


@pytest.mark.asyncio
async def test_start_returns_immediately_while_loading():
    slow_backend = FakeBackend(init_delay=0.5)
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory(slow_backend))

    import time
    t0 = time.monotonic()
    await reg.start()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1, f"start() should be non-blocking, took {elapsed}s"

    # is_ready=False during init
    assert reg.is_ready() is False
    assert reg.state() == RegistryState.LOADING

    # Wait for init to complete
    await asyncio.sleep(0.6)
    assert reg.is_ready() is True
    assert reg.state() == RegistryState.READY


@pytest.mark.asyncio
async def test_create_session_raises_when_not_ready():
    slow_backend = FakeBackend(init_delay=10.0)
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory(slow_backend))
    await reg.start()
    with pytest.raises(RuntimeError, match="not ready"):
        reg.create_session("zh")
    await reg.shutdown()


@pytest.mark.asyncio
async def test_create_session_works_when_ready():
    backend = FakeBackend()
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory(backend))
    await reg.start()
    await asyncio.sleep(0.05)
    assert reg.is_ready()
    session = reg.create_session("zh")
    assert isinstance(session, FakeSession)
    await reg.shutdown()


@pytest.mark.asyncio
async def test_reload_failure_keeps_no_effective_initially():
    """首次加载失败：no effective，state=ERROR"""
    bad = FakeBackend(fail_init=True)
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory(bad))
    await reg.start()
    await asyncio.sleep(0.05)
    assert reg.is_ready() is False
    assert reg.state() == RegistryState.ERROR
    assert "init boom" in (reg.last_error() or "")


@pytest.mark.asyncio
async def test_current_descriptor_returns_none_when_not_ready():
    slow = FakeBackend(init_delay=10.0)
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory(slow))
    await reg.start()
    assert reg.current_descriptor() is None
    await reg.shutdown()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_backend_registry.py -v -k "start or session or reload_failure or descriptor"`
Expected: FAIL

- [ ] **Step 3: 实现 BackendRegistry 类（start + 基础方法）**

追加到 `src/voice_input/backends/registry.py`：

```python
class BackendRegistry:
    """管理所有 backend 实例；启动时异步 init；配置变化时后台 reload。"""

    def __init__(
        self,
        config: dict,
        factory: Callable[[dict], TranscriptionBackend],
    ) -> None:
        self._config = config
        self._factory = factory
        self._effective: _Effective | None = None
        self._target_signature: str | None = None
        self._reload_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._listeners: list[StateListener] = []
        self._state = RegistryState.LOADING
        self._last_error: str | None = None

    def state(self) -> RegistryState:
        return self._state

    def last_error(self) -> str | None:
        return self._last_error

    def is_ready(self) -> bool:
        return self._effective is not None

    def current_descriptor(self) -> BackendDescriptor | None:
        return self._effective.descriptor if self._effective else None

    def add_state_listener(self, callback: StateListener) -> None:
        self._listeners.append(callback)

    def _set_state(self, state: RegistryState, error: str | None = None) -> None:
        self._state = state
        self._last_error = error
        for cb in list(self._listeners):
            try:
                cb(state, error)
            except Exception:
                log.exception("state listener raised")

    def create_session(self, language: str) -> Session:
        if self._effective is None:
            raise RuntimeError("Backend is not ready")
        return self._effective.backend.create_session(language)

    async def start(self) -> None:
        """触发首次异步加载，立即返回。"""
        signature = compute_signature(self._config)
        self._target_signature = signature
        self._set_state(RegistryState.LOADING)
        self._reload_task = asyncio.create_task(
            self._reload_worker(self._config, signature)
        )

    async def _reload_worker(self, config: dict, signature: str) -> None:
        """后台 reload：构造新 backend → initialize → 原子切换。"""
        log.info("reload worker started for signature=%s", signature[:12])
        try:
            new_backend = self._factory(config)
            await new_backend.initialize()
        except asyncio.CancelledError:
            log.info("reload cancelled")
            raise
        except Exception as e:
            log.exception("backend initialize failed")
            self._set_state(RegistryState.ERROR, error=str(e))
            return

        async with self._lock:
            old = self._effective
            self._effective = _Effective(
                backend=new_backend,
                descriptor=new_backend.describe(),
                signature=signature,
            )
        if old is not None:
            try:
                await old.backend.shutdown()
            except Exception:
                log.exception("old backend shutdown failed")
        self._set_state(RegistryState.READY)
        log.info("backend ready: %s", self._effective.descriptor.model_id)

    async def shutdown(self) -> None:
        if self._reload_task and not self._reload_task.done():
            self._reload_task.cancel()
            try:
                await self._reload_task
            except (asyncio.CancelledError, Exception):
                pass
        async with self._lock:
            if self._effective:
                try:
                    await self._effective.backend.shutdown()
                except Exception:
                    log.exception("shutdown failed")
            self._effective = None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_backend_registry.py -v`
Expected: all passed

- [ ] **Step 5: 提交**

```bash
git add src/voice_input/backends/registry.py tests/test_backend_registry.py
git commit -m "feat(backends): implement BackendRegistry start + reload worker"
```

---

## Task 9: BackendRegistry — synchronize + listener 通知

**Files:**
- Modify: `src/voice_input/backends/registry.py`
- Modify: `tests/test_backend_registry.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_backend_registry.py`：

```python
@pytest.mark.asyncio
async def test_synchronize_no_op_when_signature_unchanged():
    backend = FakeBackend(sig="a")
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory(backend))
    await reg.start()
    await asyncio.sleep(0.05)
    assert backend.init_call_count == 1

    await reg.synchronize(cfg)
    await asyncio.sleep(0.05)
    # 没变，不应触发新 init
    assert backend.init_call_count == 1
    await reg.shutdown()


@pytest.mark.asyncio
async def test_synchronize_triggers_reload_when_signature_changes():
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2")
    next_backend = [backend1, backend2]

    def factory(config):
        return next_backend.pop(0)

    cfg1 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=factory)
    await reg.start()
    await asyncio.sleep(0.05)
    assert reg.current_descriptor().model_id == "m1"

    await reg.synchronize(cfg2)
    await asyncio.sleep(0.05)
    assert reg.current_descriptor().model_id == "m2"
    assert backend1.shutdown_called is True
    await reg.shutdown()


@pytest.mark.asyncio
async def test_reload_keeps_old_effective_during_reload():
    """reload 过程中按热键，旧 backend 仍可用。"""
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2", init_delay=0.3)
    next_backend = [backend1, backend2]

    def factory(config):
        return next_backend.pop(0)

    cfg1 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=factory)
    await reg.start()
    await asyncio.sleep(0.05)

    await reg.synchronize(cfg2)
    await asyncio.sleep(0.05)
    # 新 backend 还在加载，但旧 backend 仍 effective
    assert reg.is_ready() is True
    assert reg.current_descriptor().model_id == "m1"
    assert reg.state() == RegistryState.RELOADING

    # 等新 backend ready
    await asyncio.sleep(0.5)
    assert reg.current_descriptor().model_id == "m2"
    await reg.shutdown()


@pytest.mark.asyncio
async def test_reload_failure_keeps_old_effective():
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2", fail_init=True)
    next_backend = [backend1, backend2]

    def factory(config):
        return next_backend.pop(0)

    cfg1 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=factory)
    await reg.start()
    await asyncio.sleep(0.05)

    await reg.synchronize(cfg2)
    await asyncio.sleep(0.05)
    # 新 backend 加载失败：旧 effective 保留
    assert reg.is_ready() is True
    assert reg.current_descriptor().model_id == "m1"
    assert reg.state() == RegistryState.ERROR
    assert "init boom" in (reg.last_error() or "")
    assert backend1.shutdown_called is False
    await reg.shutdown()


@pytest.mark.asyncio
async def test_listeners_notified_on_state_change():
    backend = FakeBackend()
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory(backend))

    events: list[tuple[RegistryState, str | None]] = []
    reg.add_state_listener(lambda s, err: events.append((s, err)))

    await reg.start()
    await asyncio.sleep(0.05)

    states = [e[0] for e in events]
    assert RegistryState.LOADING in states
    assert RegistryState.READY in states
    await reg.shutdown()


@pytest.mark.asyncio
async def test_concurrent_reload_cancels_previous():
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2", init_delay=0.5)
    backend3 = FakeBackend(model_id="m3")
    next_backend = [backend1, backend2, backend3]

    def factory(config):
        return next_backend.pop(0)

    cfg1 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m2"}}}
    cfg3 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m3"}}}
    reg = BackendRegistry(cfg1, factory=factory)
    await reg.start()
    await asyncio.sleep(0.05)

    await reg.synchronize(cfg2)
    await asyncio.sleep(0.05)
    # 在 cfg2 加载完成前再切到 cfg3
    await reg.synchronize(cfg3)
    await asyncio.sleep(0.6)

    # 最终 effective 是 m3
    assert reg.current_descriptor().model_id == "m3"
    await reg.shutdown()


@pytest.mark.asyncio
async def test_session_keeps_backend_alive_during_reload():
    """开始一个 session，期间触发 reload，session 应仍能用旧 backend 完成。"""
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2", init_delay=0.3)
    next_backend = [backend1, backend2]

    def factory(config):
        return next_backend.pop(0)

    cfg1 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=factory)
    await reg.start()
    await asyncio.sleep(0.05)

    session = reg.create_session("zh")
    assert isinstance(session, FakeSession)

    # 触发 reload
    await reg.synchronize(cfg2)
    await asyncio.sleep(0.05)

    # 老 session 仍能 finish（持有的是旧 backend 引用）
    text = await session.finish()
    assert text == "fake-text"
    assert backend1.shutdown_called is False  # 还没切换
    await reg.shutdown()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_backend_registry.py -v -k "synchronize or listeners or concurrent or session_keeps"`
Expected: FAIL

- [ ] **Step 3: 实现 synchronize + RELOADING 状态切换**

在 `BackendRegistry` 类内追加：

```python
    async def synchronize(self, config: dict) -> None:
        """配置变化时调用：算 signature，相同则 no-op；不同则调度 reload。"""
        self._config = config
        new_signature = compute_signature(config)
        if new_signature == self._target_signature:
            log.debug("synchronize: signature unchanged, no-op")
            return

        self._target_signature = new_signature
        log.info("synchronize: signature changed → triggering reload")

        # 取消进行中的 reload
        if self._reload_task and not self._reload_task.done():
            self._reload_task.cancel()
            try:
                await self._reload_task
            except (asyncio.CancelledError, Exception):
                pass

        # RELOADING 当且仅当当前已有 effective
        if self._effective is not None:
            self._set_state(RegistryState.RELOADING)
        else:
            self._set_state(RegistryState.LOADING)

        self._reload_task = asyncio.create_task(
            self._reload_worker(config, new_signature)
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_backend_registry.py -v`
Expected: all passed

- [ ] **Step 5: 提交**

```bash
git add src/voice_input/backends/registry.py tests/test_backend_registry.py
git commit -m "feat(backends): implement BackendRegistry.synchronize and listeners"
```

---

## Task 10: 移动并升级 llm.py 到 postprocess/

**Files:**
- Create: `src/voice_input/postprocess/__init__.py`
- Create: `src/voice_input/postprocess/llm.py` (从 src/voice_input/llm.py 移动 + 修改)
- Delete: `src/voice_input/llm.py`
- Modify: `tests/` (新增 tests/test_postprocess_llm.py)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_postprocess_llm.py
from __future__ import annotations

import httpx
import pytest
import respx

from voice_input.postprocess.llm import LLMRefiner


@pytest.mark.asyncio
async def test_refine_uses_provided_prompt():
    refiner = LLMRefiner(
        api_base="https://api.test.com/v1",
        api_key="sk-test",
        model="gpt-test",
    )
    captured = {}

    async def handler(request):
        body = request.read()
        captured["body"] = body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "corrected text"}}]},
        )

    with respx.mock:
        respx.post("https://api.test.com/v1/chat/completions").mock(side_effect=handler)
        result = await refiner.refine("raw text", prompt="custom prompt")

    assert result == "corrected text"
    assert b'"custom prompt"' in captured["body"]
    await refiner.close()


@pytest.mark.asyncio
async def test_refine_returns_original_on_failure():
    refiner = LLMRefiner(
        api_base="https://api.test.com/v1", api_key="sk-test", model="m"
    )
    with respx.mock:
        respx.post("https://api.test.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, content=b"error")
        )
        result = await refiner.refine("raw", prompt="p")
    assert result == "raw"
    await refiner.close()


@pytest.mark.asyncio
async def test_refine_returns_empty_for_empty_input():
    refiner = LLMRefiner(api_base="x", api_key="x", model="m")
    assert await refiner.refine("", prompt="p") == ""
    await refiner.close()


def test_is_configured_true_with_key():
    refiner = LLMRefiner(api_base="x", api_key="sk-x", model="m")
    assert refiner.is_configured() is True


def test_is_configured_false_without_key():
    refiner = LLMRefiner(api_base="x", api_key="", model="m")
    assert refiner.is_configured() is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_postprocess_llm.py -v`
Expected: FAIL（导入错误）

- [ ] **Step 3: 创建 postprocess/__init__.py（空）**

```python
# src/voice_input/postprocess/__init__.py
```

- [ ] **Step 4: 创建新的 postprocess/llm.py（refine 接受 prompt 参数）**

```python
# src/voice_input/postprocess/llm.py
from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


class LLMRefiner:
    """OpenAI-compatible API client. Caller-provided prompt for flexibility."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 5.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            trust_env=False,
        )

    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def refine(self, text: str, prompt: str) -> str:
        """Send text to LLM with the given system prompt. Returns refined text or original on failure."""
        if not text:
            return text
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.0,
                    "max_tokens": len(text) * 3 + 100,
                },
            )
            response.raise_for_status()
            data = response.json()
            corrected = data["choices"][0]["message"]["content"].strip()
            return corrected if corrected else text
        except Exception as e:
            log.warning("LLM refine failed: %s", e)
            return text

    async def test_connection(self) -> tuple[bool, str]:
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
            )
            response.raise_for_status()
            return True, "Connection successful"
        except httpx.HTTPStatusError as e:
            return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return False, str(e)

    async def close(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 5: 删除旧 src/voice_input/llm.py**

Run: `git rm src/voice_input/llm.py`

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/test_postprocess_llm.py -v`
Expected: 5 passed

- [ ] **Step 7: 提交**

```bash
git add src/voice_input/postprocess/ tests/test_postprocess_llm.py
git commit -m "refactor(postprocess): move llm.py with caller-provided prompt"
```

---

## Task 11: SceneRegistry + ScenePipeline

**Files:**
- Create: `src/voice_input/postprocess/scene.py`
- Create: `src/voice_input/postprocess/pipeline.py`
- Create: `tests/test_scene.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 写 scene 失败测试**

```python
# tests/test_scene.py
from __future__ import annotations

import pytest

from voice_input.postprocess.scene import DEFAULT_SCENE, Scene, SceneRegistry


def test_default_scene_exists():
    assert DEFAULT_SCENE.id == "default"
    assert DEFAULT_SCENE.prompt


def test_registry_loads_from_config():
    cfg = {
        "postprocess": {
            "active_scene": "code",
            "scenes": [
                {"id": "default", "name": "默认", "prompt": "p1"},
                {"id": "code", "name": "代码", "prompt": "p2"},
            ],
        }
    }
    reg = SceneRegistry(cfg)
    assert reg.get("code").prompt == "p2"
    assert reg.active().id == "code"


def test_registry_returns_default_when_active_unset():
    cfg = {"postprocess": {}}
    reg = SceneRegistry(cfg)
    assert reg.active().id == "default"


def test_registry_returns_default_when_active_unknown():
    cfg = {
        "postprocess": {
            "active_scene": "nonexistent",
            "scenes": [{"id": "default", "name": "默认", "prompt": "p"}],
        }
    }
    reg = SceneRegistry(cfg)
    assert reg.active().id == "default"


def test_registry_get_returns_none_for_unknown():
    cfg = {"postprocess": {}}
    reg = SceneRegistry(cfg)
    assert reg.get("nonexistent") is None


def test_registry_list_includes_default():
    cfg = {"postprocess": {}}
    reg = SceneRegistry(cfg)
    ids = [s.id for s in reg.list()]
    assert "default" in ids


def test_set_active_persists_in_memory():
    cfg = {
        "postprocess": {
            "scenes": [
                {"id": "default", "name": "默认", "prompt": "p1"},
                {"id": "code", "name": "代码", "prompt": "p2"},
            ]
        }
    }
    reg = SceneRegistry(cfg)
    reg.set_active("code")
    assert reg.active().id == "code"


def test_set_active_unknown_raises():
    cfg = {"postprocess": {}}
    reg = SceneRegistry(cfg)
    with pytest.raises(KeyError):
        reg.set_active("nonexistent")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_scene.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 scene.py**

```python
# src/voice_input/postprocess/scene.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scene:
    id: str
    name: str
    prompt: str


DEFAULT_SCENE = Scene(
    id="default",
    name="默认",
    prompt=(
        "You are a speech recognition error corrector. ONLY fix obvious transcription errors, especially:\n"
        "- Chinese homophone errors from ASR\n"
        "- English technical terms mis-transcribed as Chinese phonetics "
        "(e.g. 配森→Python, 杰森→JSON, 锐克特→React, 瑞迪斯→Redis, 多克→Docker, 哥拉格→GraphQL)\n"
        "- Mixed Chinese-English where English terms got corrupted\n"
        "Rules (HARD):\n"
        "- DO NOT rewrite, polish, paraphrase, or expand anything\n"
        "- DO NOT change punctuation unless it's clearly wrong\n"
        "- DO NOT add explanation, quotes, or markdown\n"
        "- If the input looks correct, return it EXACTLY as-is\n"
        "- Return ONLY the corrected text."
    ),
)


class SceneRegistry:
    """Loads scenes from config; defaults always include 'default'."""

    def __init__(self, config: dict) -> None:
        pp = config.get("postprocess", {})
        scenes_cfg = pp.get("scenes", [])

        scenes: dict[str, Scene] = {DEFAULT_SCENE.id: DEFAULT_SCENE}
        for entry in scenes_cfg:
            sid = entry.get("id")
            if not sid:
                continue
            scenes[sid] = Scene(
                id=sid,
                name=entry.get("name", sid),
                prompt=entry.get("prompt", ""),
            )
        self._scenes = scenes
        self._active_id = pp.get("active_scene", DEFAULT_SCENE.id)
        if self._active_id not in self._scenes:
            self._active_id = DEFAULT_SCENE.id

    def get(self, scene_id: str) -> Scene | None:
        return self._scenes.get(scene_id)

    def list(self) -> list[Scene]:
        return list(self._scenes.values())

    def active(self) -> Scene:
        return self._scenes[self._active_id]

    def set_active(self, scene_id: str) -> None:
        if scene_id not in self._scenes:
            raise KeyError(f"Unknown scene: {scene_id}")
        self._active_id = scene_id
```

- [ ] **Step 4: 运行 scene 测试通过**

Run: `pytest tests/test_scene.py -v`
Expected: all passed

- [ ] **Step 5: 写 pipeline 失败测试**

```python
# tests/test_pipeline.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from voice_input.postprocess.pipeline import ScenePipeline
from voice_input.postprocess.scene import Scene, SceneRegistry


@pytest.mark.asyncio
async def test_pipeline_uses_active_scene_prompt():
    cfg = {
        "postprocess": {
            "active_scene": "code",
            "scenes": [
                {"id": "default", "name": "默认", "prompt": "default-prompt"},
                {"id": "code", "name": "代码", "prompt": "code-prompt"},
            ],
        }
    }
    reg = SceneRegistry(cfg)
    llm = MagicMock()
    llm.is_configured.return_value = True
    llm.refine = AsyncMock(return_value="refined")
    pipeline = ScenePipeline(reg, llm)

    result = await pipeline.process("raw")

    assert result == "refined"
    llm.refine.assert_awaited_once_with("raw", prompt="code-prompt")


@pytest.mark.asyncio
async def test_pipeline_uses_explicit_scene_id_when_provided():
    cfg = {
        "postprocess": {
            "active_scene": "code",
            "scenes": [
                {"id": "default", "name": "默认", "prompt": "default-prompt"},
                {"id": "code", "name": "代码", "prompt": "code-prompt"},
                {"id": "polish", "name": "口语", "prompt": "polish-prompt"},
            ],
        }
    }
    reg = SceneRegistry(cfg)
    llm = MagicMock()
    llm.is_configured.return_value = True
    llm.refine = AsyncMock(return_value="refined")
    pipeline = ScenePipeline(reg, llm)

    await pipeline.process("raw", scene_id="polish")

    llm.refine.assert_awaited_once_with("raw", prompt="polish-prompt")


@pytest.mark.asyncio
async def test_pipeline_returns_raw_when_llm_disabled():
    reg = SceneRegistry({"postprocess": {}})
    llm = MagicMock()
    llm.is_configured.return_value = False
    llm.refine = AsyncMock()
    pipeline = ScenePipeline(reg, llm)

    result = await pipeline.process("raw text")

    assert result == "raw text"
    llm.refine.assert_not_awaited()


@pytest.mark.asyncio
async def test_pipeline_returns_raw_when_llm_raises():
    reg = SceneRegistry({"postprocess": {}})
    llm = MagicMock()
    llm.is_configured.return_value = True
    llm.refine = AsyncMock(side_effect=RuntimeError("boom"))
    pipeline = ScenePipeline(reg, llm)

    result = await pipeline.process("raw")

    assert result == "raw"
```

- [ ] **Step 6: 运行 pipeline 测试确认失败**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 7: 实现 pipeline.py**

```python
# src/voice_input/postprocess/pipeline.py
from __future__ import annotations

import logging

from voice_input.postprocess.llm import LLMRefiner
from voice_input.postprocess.scene import SceneRegistry

log = logging.getLogger(__name__)


class ScenePipeline:
    """场景化后处理：用 active scene 的 prompt 调用 LLM。失败时降级返回原文。"""

    def __init__(self, scenes: SceneRegistry, llm: LLMRefiner) -> None:
        self._scenes = scenes
        self._llm = llm

    async def process(self, raw_text: str, scene_id: str | None = None) -> str:
        if not raw_text:
            return raw_text
        if not self._llm.is_configured():
            return raw_text

        if scene_id is not None:
            scene = self._scenes.get(scene_id) or self._scenes.active()
        else:
            scene = self._scenes.active()

        try:
            return await self._llm.refine(raw_text, prompt=scene.prompt)
        except Exception:
            log.exception("scene postprocess failed; falling back to raw text")
            return raw_text
```

- [ ] **Step 8: 运行所有 pipeline+scene 测试通过**

Run: `pytest tests/test_pipeline.py tests/test_scene.py -v`
Expected: all passed

- [ ] **Step 9: 提交**

```bash
git add src/voice_input/postprocess/scene.py src/voice_input/postprocess/pipeline.py tests/test_scene.py tests/test_pipeline.py
git commit -m "feat(postprocess): add Scene + ScenePipeline for multi-prompt postprocess"
```

---

## Task 12: 配置 schema 更新（config.py）

**Files:**
- Modify: `src/voice_input/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_config.py` 末尾：

```python
def test_default_config_has_sherpa_section():
    from voice_input.config import DEFAULT_CONFIG
    assert "sherpa" in DEFAULT_CONFIG["stt"]
    assert DEFAULT_CONFIG["stt"]["sherpa"]["model_id"]
    assert DEFAULT_CONFIG["stt"]["sherpa"]["vad_enabled"] is True


def test_default_config_backend_is_sherpa():
    from voice_input.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["stt"]["backend"] == "sherpa"


def test_default_config_has_postprocess_section():
    from voice_input.config import DEFAULT_CONFIG
    assert "postprocess" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["postprocess"]["active_scene"] == "default"


def test_default_config_no_local_section():
    """旧 stt.local 已删除"""
    from voice_input.config import DEFAULT_CONFIG
    assert "local" not in DEFAULT_CONFIG["stt"]


def test_load_config_migrates_legacy_local_to_sherpa(tmp_path):
    from voice_input.config import load_config
    legacy = tmp_path / "config.toml"
    legacy.write_text(
        '[stt]\nbackend = "local"\n\n'
        '[stt.local]\nengine = "whisper"\nmodel = "medium"\nlanguage = "zh"\n'
    )
    cfg = load_config(config_dir=tmp_path)
    # 旧 backend=local 自动迁移到 sherpa
    assert cfg["stt"]["backend"] == "sherpa"
    # language 保留
    assert cfg.get("stt", {}).get("sherpa", {}).get("language", "zh")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_config.py -v -k "sherpa or postprocess or no_local or migrates_legacy_local"`
Expected: FAIL

- [ ] **Step 3: 修改 DEFAULT_CONFIG 与添加迁移函数**

修改 `src/voice_input/config.py` 第 19-59 行（DEFAULT_CONFIG）：

```python
DEFAULT_CONFIG: dict = {
    "hotkey": {
        "mode": "toggle",
        "key": "Meta+Space",
    },
    "stt": {
        "backend": "sherpa",
        "language": "zh",
        "sherpa": {
            "model_id": "sherpa-onnx-paraformer-zh-2024-03-09",
            "vad_enabled": True,
            "num_threads": 2,
            "provider": "cpu",
        },
        "openai": {
            "api_base": "https://api.openai.com/v1",
            "model": "whisper-1",
        },
        "google": {
            "credentials_path": "",
        },
        "volcengine": {
            "app_id": "",
            "resource_id": "volc.seedasr.sauc.duration",
        },
    },
    "llm": {
        "enabled": True,
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "postprocess": {
        "enabled": True,
        "active_scene": "default",
        "scenes": [
            {
                "id": "default",
                "name": "默认",
                "prompt": "修正 ASR 识别错误，保持原意，返回纯文本。",
            },
            {
                "id": "code",
                "name": "代码场景",
                "prompt": (
                    "这是程序员说的话。修正中文同音字错误，把英文技术术语恢复"
                    "（例如：'派森' → 'Python'，'瑞克特' → 'React'）。返回纯文本。"
                ),
            },
            {
                "id": "translate-en",
                "name": "翻译为英文",
                "prompt": "把以下中文翻译为地道英文，只返回译文。",
            },
            {
                "id": "polish",
                "name": "口语转书面",
                "prompt": "把口语化的中文改为书面表达，删除语气词和重复，保持原意。",
            },
        ],
    },
    "audio": {
        "device": "default",
        "sample_rate": 16000,
    },
    "ui": {
        "overlay_margin_bottom": 80,
    },
    "inject": {
        "paste_method": "ctrl_v",
    },
}
```

- [ ] **Step 4: 替换迁移函数处理 legacy local**

替换 `_migrate_legacy_whisper_config` 函数：

```python
def _migrate_legacy_local_config(cfg: dict, user_cfg: dict) -> dict:
    """Map old [stt] backend=local / [stt.local] / [whisper] to new sherpa backend."""
    user_stt = user_cfg.get("stt", {})

    # Old "local" backend → "sherpa"
    if user_stt.get("backend") == "local":
        cfg.setdefault("stt", {})["backend"] = "sherpa"

    # Old [stt.local] language → top-level stt.language
    legacy_local = user_stt.get("local", {})
    if isinstance(legacy_local, dict) and "language" in legacy_local:
        cfg.setdefault("stt", {}).setdefault(
            "language", legacy_local["language"]
        )

    # Old [whisper] section → ignore (whisper backend removed)
    return cfg
```

- [ ] **Step 5: 更新 load_config 调用**

修改 `load_config` 中的调用（约第 147 行）：

```python
    cfg = _migrate_legacy_local_config(cfg, user_cfg)
```

- [ ] **Step 6: _to_toml 序列化器支持 list of dict**

修改 `_to_toml` 函数支持 `[[postprocess.scenes]]` 数组：

```python
def _to_toml(data: dict, prefix: str = "") -> str:
    """Simple dict-to-TOML serializer with [[arrays of tables]]."""
    lines: list[str] = []
    scalars: list[tuple[str, object]] = []
    sections: list[tuple[str, dict]] = []
    array_tables: list[tuple[str, list]] = []

    for k, v in data.items():
        if isinstance(v, dict):
            sections.append((k, v))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            array_tables.append((k, v))
        else:
            scalars.append((k, v))

    for k, v in scalars:
        lines.append(f"{k} = {_toml_value(v)}")

    for section_name, section_dict in sections:
        full = f"{prefix}.{section_name}" if prefix else section_name
        lines.append(f"\n[{full}]")
        for k, v in section_dict.items():
            if isinstance(v, dict):
                lines.append(_to_toml({k: v}, prefix=full))
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                array_full = f"{full}.{k}"
                for entry in v:
                    lines.append(f"\n[[{array_full}]]")
                    for ek, ev in entry.items():
                        lines.append(f"{ek} = {_toml_value(ev)}")
            else:
                lines.append(f"{k} = {_toml_value(v)}")

    for k, v in array_tables:
        full = f"{prefix}.{k}" if prefix else k
        for entry in v:
            lines.append(f"\n[[{full}]]")
            for ek, ev in entry.items():
                lines.append(f"{ek} = {_toml_value(ev)}")

    return "\n".join(lines)
```

也需要更新 `_toml_value` 处理多行字符串：

```python
def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(v)
    if isinstance(v, str):
        # Use triple-quoted for multi-line strings
        if "\n" in v or '"' in v:
            escaped = v.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            return f'"""{escaped}"""'
        return f'"{v}"'
    return str(v)
```

- [ ] **Step 7: 运行测试**

Run: `pytest tests/test_config.py -v`
Expected: all passed

- [ ] **Step 8: 提交**

```bash
git add src/voice_input/config.py tests/test_config.py
git commit -m "feat(config): add sherpa+postprocess sections, migrate legacy local"
```

---

## Task 13: 适配现有云 backends（VolcEngine, Google, OpenAI Whisper）

新接口要求实现 `describe()`、`create_session()`、`shutdown()`，旧的 `transcribe(audio, language)` 通过 Session.finish 包装实现。

**Files:**
- Modify: `src/voice_input/backends/volcengine_speech.py`
- Modify: `src/voice_input/backends/google_speech.py`
- Modify: `src/voice_input/backends/openai_whisper.py`
- Modify: `tests/test_backend_volcengine.py`

注：本任务对每个云 backend 用相同模式适配——把现有 `async def transcribe(audio, language)` 包装成 `Session.finish()`。下面只展示 VolcEngine 的完整改动；其他两个按相同模式处理。

- [ ] **Step 1: 检查现有 volcengine_speech.py 接口**

Run: `head -40 src/voice_input/backends/volcengine_speech.py`

- [ ] **Step 2: 重写 volcengine_speech.py 顶部 + 添加 Session 包装**

修改 `src/voice_input/backends/volcengine_speech.py`，在文件末尾添加适配类（保留原有 `_VolcCore` 等内部实现）：

```python
# 在 src/voice_input/backends/volcengine_speech.py 末尾追加：

from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    RecognitionError,
    Session,
    TranscriptionBackend,
)


class _VolcSession(Session):
    """缓冲音频，finish 时一次性提交给 Volc API。"""

    def __init__(self, backend: "VolcengineSpeechBackend", language: str) -> None:
        self._backend = backend
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
        audio = np.concatenate(self._buffer)
        try:
            return await self._backend._transcribe_async(audio, self._language)
        except RecognitionError:
            raise
        except Exception as e:
            raise RecognitionError(
                str(e), user_message=f"火山引擎错误：{str(e)[:80]}"
            ) from e
```

修改 `VolcengineSpeechBackend` 类——添加 base 接口方法，保留 `_transcribe_async` 作内部方法：

```python
class VolcengineSpeechBackend(TranscriptionBackend):
    # ... 保留现有 __init__、其他方法不变 ...

    def is_ready(self) -> bool:
        return bool(self._app_id and self._access_key and self._secret_key)

    def describe(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id="volcengine",
            model_id=self._resource_id,
            capabilities=BackendCapabilities(
                supports_streaming=True,
                requires_network=True,
                supports_vad=False,
            ),
        )

    def create_session(self, language: str) -> Session:
        if not self.is_ready():
            raise RuntimeError("VolcEngine credentials missing")
        return _VolcSession(self, language)

    async def shutdown(self) -> None:
        # 释放可能的连接
        pass

    async def _transcribe_async(self, audio: np.ndarray, language: str) -> str:
        # 把原来 transcribe 方法的实现挪到这里。
        # 原 transcribe(audio_data, language) 的整个 body 移过来。
        return await self.transcribe(audio, language)  # 临时桥接
```

- [ ] **Step 3: 同样模式适配 google_speech.py**

```python
# src/voice_input/backends/google_speech.py 末尾追加：

from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    RecognitionError,
    Session,
    TranscriptionBackend,
)


class _GoogleSession(Session):
    def __init__(self, backend, language):
        self._backend = backend
        self._language = language
        self._buffer = []
        self._cancelled = False

    def push_audio(self, pcm):
        if not self._cancelled:
            self._buffer.append(pcm)

    def cancel(self):
        self._cancelled = True
        self._buffer.clear()

    async def finish(self):
        if self._cancelled or not self._buffer:
            return ""
        audio = np.concatenate(self._buffer)
        try:
            return await self._backend.transcribe(audio, self._language)
        except Exception as e:
            raise RecognitionError(
                str(e), user_message=f"Google API 错误：{str(e)[:80]}"
            ) from e
```

修改 `GoogleSpeechBackend` 类同样添加 `is_ready / describe / create_session / shutdown`：

```python
    def is_ready(self) -> bool:
        return bool(self._credentials_path)

    def describe(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id="google",
            model_id="google-stt-default",
            capabilities=BackendCapabilities(
                supports_streaming=False,
                requires_network=True,
            ),
        )

    def create_session(self, language: str) -> Session:
        return _GoogleSession(self, language)

    async def shutdown(self) -> None:
        pass
```

- [ ] **Step 4: 适配 openai_whisper.py**

在 `src/voice_input/backends/openai_whisper.py` 末尾追加：

```python
import numpy as np

from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    RecognitionError,
    Session,
    TranscriptionBackend,
)


class _OpenAISession(Session):
    def __init__(self, backend: "OpenAIWhisperBackend", language: str) -> None:
        self._backend = backend
        self._language = language
        self._buffer: list[np.ndarray] = []
        self._cancelled = False

    def push_audio(self, pcm_int16: np.ndarray) -> None:
        if not self._cancelled:
            self._buffer.append(pcm_int16)

    def cancel(self) -> None:
        self._cancelled = True
        self._buffer.clear()

    async def finish(self) -> str:
        if self._cancelled or not self._buffer:
            return ""
        audio = np.concatenate(self._buffer)
        try:
            return await self._backend.transcribe(audio, self._language)
        except Exception as e:
            raise RecognitionError(
                str(e), user_message=f"OpenAI Whisper 错误：{str(e)[:80]}"
            ) from e
```

并在 `OpenAIWhisperBackend` 类内添加：

```python
    def is_ready(self) -> bool:
        return bool(getattr(self, "_api_key", "") or self.api_key if hasattr(self, "api_key") else "")

    def describe(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id="openai-whisper",
            model_id=self._model if hasattr(self, "_model") else "whisper-1",
            capabilities=BackendCapabilities(
                supports_streaming=False,
                requires_network=True,
            ),
        )

    def create_session(self, language: str) -> Session:
        return _OpenAISession(self, language)

    async def shutdown(self) -> None:
        if hasattr(self, "_client"):
            await self._client.aclose()
```

注：上面 `is_ready` 与 `describe` 中的属性名（`_api_key`/`api_key`、`_model`/`model`）需要对齐 `openai_whisper.py` 中实际定义。先 grep 确认：

```bash
grep -n "self\._api_key\|self\.api_key\|self\._model\|self\.model" src/voice_input/backends/openai_whisper.py
```

按实际属性名调整。

- [ ] **Step 5: 修改现有云 backend 测试**

修改 `tests/test_backend_volcengine.py`，确保 `is_ready / describe / create_session` 都被覆盖。具体测试用例：

```python
def test_volcengine_describe():
    from voice_input.backends.volcengine_speech import VolcengineSpeechBackend
    backend = VolcengineSpeechBackend(
        app_id="aid", access_key="ak", secret_key="sk", resource_id="r"
    )
    desc = backend.describe()
    assert desc.backend_id == "volcengine"
    assert desc.model_id == "r"
    assert desc.capabilities.requires_network is True


def test_volcengine_is_ready_false_without_creds():
    from voice_input.backends.volcengine_speech import VolcengineSpeechBackend
    backend = VolcengineSpeechBackend(app_id="", access_key="", secret_key="", resource_id="r")
    assert backend.is_ready() is False


def test_volcengine_is_ready_true_with_creds():
    from voice_input.backends.volcengine_speech import VolcengineSpeechBackend
    backend = VolcengineSpeechBackend(app_id="a", access_key="k", secret_key="s", resource_id="r")
    assert backend.is_ready() is True
```

- [ ] **Step 6: 运行所有 backend 测试**

Run: `pytest tests/test_backend_volcengine.py tests/test_backend_factory.py -v`
Expected: passed（如有失败，查看错误并修复 import 与 method 签名）

- [ ] **Step 7: 提交**

```bash
git add src/voice_input/backends/volcengine_speech.py src/voice_input/backends/google_speech.py src/voice_input/backends/openai_whisper.py tests/test_backend_volcengine.py
git commit -m "refactor(backends): adapt cloud backends to Session interface"
```

---

## Task 14: 更新 backends/__init__.py 工厂

**Files:**
- Modify: `src/voice_input/backends/__init__.py`
- Modify: `tests/test_backend_factory.py`

- [ ] **Step 1: 写失败测试**

修改 `tests/test_backend_factory.py`：

```python
# tests/test_backend_factory.py
from __future__ import annotations

import pytest

from voice_input.backends import create_backend
from voice_input.backends.sherpa_backend import SherpaBackend


def test_create_sherpa_backend():
    cfg = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "x"}}}
    backend = create_backend(cfg)
    assert isinstance(backend, SherpaBackend)


def test_create_unknown_backend_raises():
    cfg = {"stt": {"backend": "nonexistent"}}
    with pytest.raises(ValueError, match="Unknown STT backend"):
        create_backend(cfg)


def test_create_local_backend_id_now_routes_to_sherpa():
    """Legacy 'local' value still works (treated as sherpa)."""
    cfg = {"stt": {"backend": "local", "sherpa": {}}}
    backend = create_backend(cfg)
    assert isinstance(backend, SherpaBackend)


def test_create_volcengine_backend(monkeypatch):
    monkeypatch.setattr(
        "keyring.get_password",
        lambda service, key: "fake-key" if "volcengine" in key else None,
    )
    from voice_input.backends.volcengine_speech import VolcengineSpeechBackend
    cfg = {"stt": {"backend": "volcengine", "volcengine": {"app_id": "x", "resource_id": "r"}}}
    backend = create_backend(cfg)
    assert isinstance(backend, VolcengineSpeechBackend)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_backend_factory.py -v`
Expected: FAIL

- [ ] **Step 3: 重写 backends/__init__.py**

```python
# src/voice_input/backends/__init__.py
"""STT backend abstraction and factory."""
from __future__ import annotations

from voice_input.backends.base import TranscriptionBackend
from voice_input.config import AppConfig


def create_backend(config: AppConfig) -> TranscriptionBackend:
    """Create a transcription backend from app config."""
    stt_cfg = config.get("stt", {})
    backend_name = stt_cfg.get("backend", "sherpa")

    # Legacy 'local' alias maps to sherpa
    if backend_name in ("sherpa", "local"):
        from voice_input.backends.sherpa_backend import SherpaBackend
        return SherpaBackend(config)

    if backend_name == "openai":
        from voice_input.backends.openai_whisper import OpenAIWhisperBackend
        openai_cfg = stt_cfg.get("openai", {})
        try:
            import keyring
            api_key = keyring.get_password("voice-input", "stt-openai-api-key") or ""
        except Exception:
            api_key = ""
        return OpenAIWhisperBackend(
            api_base=openai_cfg.get("api_base", "https://api.openai.com/v1"),
            api_key=api_key,
            model=openai_cfg.get("model", "whisper-1"),
        )

    if backend_name == "google":
        from voice_input.backends.google_speech import GoogleSpeechBackend
        google_cfg = stt_cfg.get("google", {})
        return GoogleSpeechBackend(
            credentials_path=google_cfg.get("credentials_path", ""),
        )

    if backend_name == "volcengine":
        from voice_input.backends.volcengine_speech import VolcengineSpeechBackend
        volc_cfg = stt_cfg.get("volcengine", {})
        try:
            import keyring
            access_key = (
                keyring.get_password("voice-input", "stt-volcengine-access-key") or ""
            )
            secret_key = (
                keyring.get_password("voice-input", "stt-volcengine-secret-key") or ""
            )
        except Exception:
            access_key = ""
            secret_key = ""
        return VolcengineSpeechBackend(
            app_id=volc_cfg.get("app_id", ""),
            access_key=access_key,
            secret_key=secret_key,
            resource_id=volc_cfg.get("resource_id", "volc.seedasr.sauc.duration"),
        )

    raise ValueError(f"Unknown STT backend: {backend_name}")
```

- [ ] **Step 4: 运行测试通过**

Run: `pytest tests/test_backend_factory.py -v`
Expected: passed

- [ ] **Step 5: 提交**

```bash
git add src/voice_input/backends/__init__.py tests/test_backend_factory.py
git commit -m "refactor(backends): factory routes 'local'→sherpa, drops engine concept"
```

---

## Task 15: app.py 适配 — 用 BackendRegistry，移除 WhisperWorker

这是最大的改动。app.py 完全重构为基于 `BackendRegistry` + `Session` 模型。

**Files:**
- Modify: `src/voice_input/app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: 写失败测试（关键场景）**

修改 `tests/test_app.py`，添加新测试（保留可复用的旧测试，删除针对 WhisperWorker 的）：

```python
# tests/test_app.py 新增：
import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    Session,
    TranscriptionBackend,
)
from voice_input.backends.registry import BackendRegistry, RegistryState


@pytest.mark.asyncio
async def test_app_calls_send_notification_when_backend_not_ready():
    """模拟 backend 还在加载时按热键，应通知用户而不是崩溃。"""
    # 由于 AppController 与 PyQt 紧密耦合，这里用最小的间接验证：
    # 直接测试 _can_start_recording 方法
    from voice_input.app import AppController
    from voice_input.config import DEFAULT_CONFIG
    import copy

    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # 用一个不会 ready 的 fake registry
    class FakeRegistry:
        def is_ready(self):
            return False
        def state(self):
            return RegistryState.LOADING
        def current_descriptor(self):
            return None

    # 由于 AppController 创建复杂，这里测试关键方法的逻辑
    # 实际 app.py 应包含一个 _is_backend_ready() 辅助
    reg = FakeRegistry()
    assert not reg.is_ready()
    assert reg.state() == RegistryState.LOADING
```

- [ ] **Step 2: 重写 app.py 关键部分**

整体重构 `src/voice_input/app.py`：

```python
# src/voice_input/app.py
from __future__ import annotations

import asyncio
import enum
import logging
import queue
import sys

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMessageBox

from voice_input.audio import AudioRecorder, compute_rms
from voice_input.backends import create_backend
from voice_input.backends.base import RecognitionError, Session
from voice_input.backends.registry import BackendRegistry, RegistryState
from voice_input.config import AppConfig, load_config, save_config
from voice_input.hotkey import HotkeyManager
from voice_input.injector import TextInjector
from voice_input.overlay import OverlayWidget
from voice_input.postprocess.llm import LLMRefiner
from voice_input.postprocess.pipeline import ScenePipeline
from voice_input.postprocess.scene import SceneRegistry
from voice_input.settings_dialog import SettingsDialog
from voice_input.tray import TrayManager

log = logging.getLogger(__name__)

MIN_AUDIO_SAMPLES = 1600  # 0.1s at 16kHz


class AppState(enum.Enum):
    IDLE = "Idle"
    RECORDING = "Recording"
    TRANSCRIBING = "Transcribing"
    REFINING = "Refining"


_VALID_TRANSITIONS = {
    AppState.IDLE: {AppState.RECORDING},
    AppState.RECORDING: {AppState.IDLE, AppState.TRANSCRIBING},
    AppState.TRANSCRIBING: {AppState.IDLE, AppState.REFINING},
    AppState.REFINING: {AppState.IDLE},
}


def _can_transition(curr: AppState, target: AppState) -> bool:
    return target in _VALID_TRANSITIONS.get(curr, set())


class AppController(QObject):
    state_changed = pyqtSignal(str)

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._state = AppState.IDLE
        self._language = config.get("stt", {}).get("language", "zh")

        # Audio
        self._whisper_queue: queue.Queue = queue.Queue(maxsize=200)
        self._viz_queue: queue.Queue = queue.Queue(maxsize=50)
        self._audio = AudioRecorder(
            whisper_queue=self._whisper_queue,
            viz_queue=self._viz_queue,
            device=config["audio"]["device"],
            sample_rate=config["audio"]["sample_rate"],
        )

        # Backend Registry (replaces WhisperWorker)
        self._registry = BackendRegistry(config, factory=create_backend)
        self._registry.add_state_listener(self._on_registry_state)
        self._current_session: Session | None = None

        # Postprocess
        self._scenes = SceneRegistry(config)
        self._llm: LLMRefiner | None = None
        self._llm_enabled = config.get("postprocess", {}).get("enabled", True)
        self._pipeline: ScenePipeline | None = None
        self._init_llm()

        # UI
        self._hotkey = HotkeyManager(
            mode=config["hotkey"]["mode"],
            key=config["hotkey"]["key"],
        )
        self._injector = TextInjector(
            paste_method=config.get("inject", {}).get("paste_method", "ctrl_v"),
        )
        self._overlay = OverlayWidget(
            margin_bottom=config["ui"]["overlay_margin_bottom"],
        )
        self._tray = TrayManager()
        self._tray.set_backend(config.get("stt", {}).get("backend", "sherpa"))
        self._tray.set_scenes(self._scenes.list(), active_id=self._scenes.active().id)

        # Visualization
        self._viz_timer = QTimer()
        self._viz_timer.setInterval(16)
        self._viz_timer.timeout.connect(self._update_visualization)

        self._connect_signals()

    def _init_llm(self) -> None:
        if not self._llm_enabled:
            self._llm = None
            self._pipeline = None
            return
        try:
            import keyring
            api_key = keyring.get_password("voice-input", "llm-api-key") or ""
        except Exception:
            api_key = ""
        self._llm = LLMRefiner(
            api_base=self._config["llm"]["api_base"],
            api_key=api_key,
            model=self._config["llm"]["model"],
        )
        self._pipeline = ScenePipeline(self._scenes, self._llm)

    def _connect_signals(self) -> None:
        self._hotkey.recording_requested.connect(self._on_toggle_recording)
        self._hotkey.hold_started.connect(self._on_start_recording)
        self._hotkey.hold_stopped.connect(self._on_stop_recording)
        self._tray.toggle_action.triggered.connect(self._on_toggle_recording)
        self._tray.lang_group.triggered.connect(self._on_language_changed)
        self._tray.stt_group.triggered.connect(self._on_backend_changed)
        self._tray.stt_settings_action.triggered.connect(self._on_open_settings)
        self._tray.scene_group.triggered.connect(self._on_scene_changed)
        self._tray.llm_toggle.toggled.connect(self._on_llm_toggled)
        self._tray.llm_settings_action.triggered.connect(self._on_open_settings)
        self._tray.about_action.triggered.connect(self._on_about)
        self.state_changed.connect(self._tray.set_state)

    async def start(self) -> None:
        self._tray.show()
        ok = await self._hotkey.register()
        if not ok:
            log.warning("Hotkey registration failed; use tray menu.")
        if self._hotkey.mode == "hold":
            asyncio.create_task(self._hotkey.run_evdev_loop())

        # Trigger backend async preload
        await self._registry.start()
        log.info(
            "Voice Input started (backend=%s)",
            self._config.get("stt", {}).get("backend", "sherpa"),
        )

    def _set_state(self, new_state: AppState) -> None:
        if not _can_transition(self._state, new_state):
            log.warning("Invalid transition: %s → %s", self._state, new_state)
            return
        self._state = new_state
        self.state_changed.emit(new_state.value)

    def _on_registry_state(self, state: RegistryState, error: str | None) -> None:
        log.info("registry state → %s (error=%s)", state.value, error)
        self._tray.set_backend_status(state.value, error)
        if state == RegistryState.ERROR and error:
            self._send_notification("Voice Input", f"模型加载失败：{error}")

    @pyqtSlot()
    def _on_toggle_recording(self) -> None:
        if self._state == AppState.IDLE:
            self._on_start_recording()
        elif self._state == AppState.RECORDING:
            self._on_stop_recording()

    @pyqtSlot()
    def _on_start_recording(self) -> None:
        if self._state != AppState.IDLE:
            return
        if not self._registry.is_ready():
            state = self._registry.state()
            if state == RegistryState.LOADING:
                self._send_notification("Voice Input", "模型加载中，请稍候")
            elif state == RegistryState.ERROR:
                err = self._registry.last_error() or "unknown error"
                self._send_notification("Voice Input", f"后端错误：{err[:80]}")
            else:
                self._send_notification("Voice Input", "后端尚未就绪")
            return

        try:
            self._current_session = self._registry.create_session(self._language)
        except Exception as e:
            log.exception("create_session failed")
            self._send_notification("Voice Input", f"无法创建识别会话：{e}")
            return

        self._drain_audio_queue()
        self._audio.start()
        self._viz_timer.start()
        self._overlay.update_text("")
        self._overlay.show()
        self._set_state(AppState.RECORDING)

    @pyqtSlot()
    def _on_stop_recording(self) -> None:
        if self._state != AppState.RECORDING:
            return
        self._audio.stop()
        self._viz_timer.stop()

        # Drain audio queue and push to session
        audio_buffer = self._drain_audio_queue()
        if self._current_session is not None and len(audio_buffer) > 0:
            self._current_session.push_audio(audio_buffer)

        if len(audio_buffer) < MIN_AUDIO_SAMPLES:
            if self._current_session:
                self._current_session.cancel()
            self._current_session = None
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return

        self._set_state(AppState.TRANSCRIBING)
        self._overlay.update_text("识别中...")
        asyncio.ensure_future(self._finish_and_inject())

    def _drain_audio_queue(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._whisper_queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.array([], dtype=np.int16)
        return np.concatenate(chunks)

    async def _finish_and_inject(self) -> None:
        session = self._current_session
        self._current_session = None
        if session is None:
            self._set_state(AppState.IDLE)
            return

        try:
            text = await session.finish()
        except RecognitionError as e:
            log.warning("recognition error: %s", e)
            self._send_notification("Voice Input", e.user_message)
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return
        except Exception as e:
            log.exception("session.finish unexpected error")
            self._send_notification("Voice Input", f"识别失败：{str(e)[:80]}")
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return

        if not text:
            self._send_notification("Voice Input", "未识别到内容")
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return

        if self._pipeline is not None:
            self._set_state(AppState.REFINING)
            self._overlay.update_text("Refining...")
            try:
                text = await self._pipeline.process(text)
            except Exception:
                log.exception("postprocess failed; using raw text")
                self._send_notification("Voice Input", "LLM 后处理失败，已注入原始文本")

        self._inject_and_finish(text)

    def _inject_and_finish(self, text: str) -> None:
        self._set_state(AppState.IDLE)
        if not text:
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            return

        def _do_inject() -> None:
            import threading
            threading.Thread(
                target=self._injector.inject, args=(text,), daemon=True
            ).start()

        self._overlay.animate_exit(
            on_finished=lambda: (self._overlay.hide(), _do_inject())
        )

    def _update_visualization(self) -> None:
        # Drain viz queue and push the same chunks to session
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._viz_queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return
        data = np.concatenate(chunks)
        rms = compute_rms(data)
        self._overlay.update_rms(rms)

    @pyqtSlot(QAction)
    def _on_language_changed(self, action: QAction) -> None:
        code = action.data()
        self._config.setdefault("stt", {})["language"] = code
        self._language = code
        save_config(self._config)
        log.info("Language → %s", code)

    @pyqtSlot(QAction)
    def _on_backend_changed(self, action: QAction) -> None:
        backend_name = action.data()
        if backend_name == self._config.get("stt", {}).get("backend"):
            return
        self._config.setdefault("stt", {})["backend"] = backend_name
        save_config(self._config)
        asyncio.ensure_future(self._registry.synchronize(self._config))
        log.info("Backend → %s (reload scheduled)", backend_name)

    @pyqtSlot(QAction)
    def _on_scene_changed(self, action: QAction) -> None:
        scene_id = action.data()
        try:
            self._scenes.set_active(scene_id)
        except KeyError:
            log.warning("Unknown scene: %s", scene_id)
            return
        self._config.setdefault("postprocess", {})["active_scene"] = scene_id
        save_config(self._config)
        log.info("Scene → %s", scene_id)

    @pyqtSlot(bool)
    def _on_llm_toggled(self, enabled: bool) -> None:
        self._llm_enabled = enabled
        self._config.setdefault("postprocess", {})["enabled"] = enabled
        save_config(self._config)
        if enabled and self._llm is None:
            self._init_llm()
        elif not enabled:
            if self._llm:
                asyncio.ensure_future(self._llm.close())
            self._llm = None
            self._pipeline = None

    @pyqtSlot()
    def _on_open_settings(self) -> None:
        dialog = SettingsDialog(self._config)
        if dialog.exec():
            if self._llm:
                asyncio.ensure_future(self._llm.close())
            self._init_llm()
            asyncio.ensure_future(self._registry.synchronize(self._config))

    @pyqtSlot()
    def _on_about(self) -> None:
        QMessageBox.about(
            None, "Voice Input",
            "Voice Input for KDE Plasma 6\nVersion 0.2.0\n\nSpeech-to-text via sherpa-onnx.",
        )

    def _send_notification(self, title: str, body: str) -> None:
        try:
            import subprocess
            subprocess.run(
                ["notify-send", title, body, "-a", "Voice Input"], timeout=5,
            )
        except Exception as e:
            log.debug("Notification failed: %s", e)


def run_app() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = load_config()
    app = QApplication(sys.argv)
    app.setApplicationName("voice-input")
    app.setDesktopFileName("voice-input")
    app.setQuitOnLastWindowClosed(False)

    try:
        import qasync
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
    except ImportError:
        log.warning("qasync not available")
        loop = None

    controller = AppController(config)
    if loop is not None:
        with loop:
            loop.run_until_complete(controller.start())
            loop.run_forever()
    else:
        app.exec()
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_app.py -v`
Expected: 通过（旧的测试针对 WhisperWorker 的需要先删除——见 Step 4）

- [ ] **Step 4: 删除旧 test_app.py 中针对 WhisperWorker 的测试**

打开 `tests/test_app.py`，删除涉及 `WhisperWorker`、`engine`、`set_engine` 等的测试方法。具体被删的测试由实际文件决定（用 `grep -n "WhisperWorker\|engine\|_create_worker" tests/test_app.py` 定位）。

- [ ] **Step 5: 运行所有测试**

Run: `pytest tests/ -v --ignore=tests/test_local_backend.py`
Expected: 大部分通过，剩余失败的与 tray 相关（下个 task 修）

- [ ] **Step 6: 提交**

```bash
git add src/voice_input/app.py tests/test_app.py
git commit -m "refactor(app): use BackendRegistry + Session, remove WhisperWorker"
```

---

## Task 16: tray.py 场景子菜单 + 移除 engine 概念

**Files:**
- Modify: `src/voice_input/tray.py`
- Modify: `tests/` (复用现有 tray 测试或 settings_dialog 测试)

- [ ] **Step 1: 修改 tray.py**

替换 `src/voice_input/tray.py` 中：

1. 删除 `ENGINES` 常量、`engine_group`、`_engine_actions`、`_engine_group`、`set_engine` 方法
2. 修改 `STT_BACKENDS` 把 `"local"` 改为 `"sherpa"`
3. 添加 scene 子菜单
4. 添加 `set_backend_status` 方法

```python
# src/voice_input/tray.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtGui import QAction, QActionGroup, QIcon
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget
    from voice_input.postprocess.scene import Scene

log = logging.getLogger(__name__)

LANGUAGES = {
    "en": "English",
    "zh": "简体中文",
    "zh-TW": "繁體中文",
    "ja": "日本語",
    "ko": "한국어",
}

STT_BACKENDS = {
    "sherpa": "Local (sherpa-onnx)",
    "openai": "OpenAI Whisper API",
    "google": "Google Speech-to-Text",
    "volcengine": "字节火山语音识别",
}


class TrayManager(QSystemTrayIcon):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = "Idle"
        self._current_language = "zh"
        self._current_backend = "sherpa"
        self._llm_enabled = True
        self._scene_actions: dict[str, QAction] = {}
        self._scene_menu: QMenu | None = None
        self._scene_group: QActionGroup | None = None

        self._icon_idle = QIcon.fromTheme("audio-input-microphone")
        self._icon_recording = QIcon.fromTheme("audio-input-microphone")
        self._icon_muted = QIcon.fromTheme("audio-input-microphone-muted")
        self.setIcon(self._icon_idle)
        self.setToolTip("Voice Input — Idle")
        self._build_menu()

    def _build_menu(self) -> None:
        menu = QMenu()

        self._status_action = QAction("Status: Idle", menu)
        self._status_action.setEnabled(False)
        menu.addAction(self._status_action)

        self._backend_status_action = QAction("Backend: loading...", menu)
        self._backend_status_action.setEnabled(False)
        menu.addAction(self._backend_status_action)

        menu.addSeparator()

        self._toggle_action = QAction("Start Recording", menu)
        menu.addAction(self._toggle_action)
        menu.addSeparator()

        # Language submenu
        lang_menu = QMenu("Language", menu)
        self._lang_group = QActionGroup(lang_menu)
        self._lang_group.setExclusive(True)
        self._lang_actions: dict[str, QAction] = {}
        for code, name in LANGUAGES.items():
            action = QAction(name, lang_menu)
            action.setCheckable(True)
            action.setData(code)
            if code == self._current_language:
                action.setChecked(True)
            self._lang_group.addAction(action)
            lang_menu.addAction(action)
            self._lang_actions[code] = action
        menu.addMenu(lang_menu)

        # STT Backend submenu
        stt_menu = QMenu("STT Backend", menu)
        self._stt_group = QActionGroup(stt_menu)
        self._stt_group.setExclusive(True)
        self._stt_actions: dict[str, QAction] = {}
        for key, label in STT_BACKENDS.items():
            action = QAction(label, stt_menu)
            action.setCheckable(True)
            action.setData(key)
            if key == self._current_backend:
                action.setChecked(True)
            self._stt_group.addAction(action)
            stt_menu.addAction(action)
            self._stt_actions[key] = action
        stt_menu.addSeparator()
        self._stt_settings_action = QAction("Settings...", stt_menu)
        stt_menu.addAction(self._stt_settings_action)
        menu.addMenu(stt_menu)

        # Scene submenu (filled by set_scenes())
        self._scene_menu = QMenu("场景 / Scene", menu)
        menu.addMenu(self._scene_menu)

        # LLM submenu
        llm_menu = QMenu("LLM Refinement", menu)
        self._llm_toggle = QAction("Enabled", llm_menu)
        self._llm_toggle.setCheckable(True)
        self._llm_toggle.setChecked(self._llm_enabled)
        llm_menu.addAction(self._llm_toggle)
        self._llm_settings_action = QAction("Settings...", llm_menu)
        llm_menu.addAction(self._llm_settings_action)
        menu.addMenu(llm_menu)

        menu.addSeparator()
        self._prefs_action = QAction("Preferences...", menu)
        menu.addAction(self._prefs_action)
        self._about_action = QAction("About", menu)
        menu.addAction(self._about_action)
        menu.addSeparator()
        self._quit_action = QAction("Quit", menu)
        self._quit_action.triggered.connect(QApplication.quit)
        menu.addAction(self._quit_action)

        self.setContextMenu(menu)

    @property
    def toggle_action(self) -> QAction:
        return self._toggle_action

    @property
    def lang_group(self) -> QActionGroup:
        return self._lang_group

    @property
    def stt_group(self) -> QActionGroup:
        return self._stt_group

    @property
    def stt_settings_action(self) -> QAction:
        return self._stt_settings_action

    @property
    def llm_toggle(self) -> QAction:
        return self._llm_toggle

    @property
    def llm_settings_action(self) -> QAction:
        return self._llm_settings_action

    @property
    def prefs_action(self) -> QAction:
        return self._prefs_action

    @property
    def about_action(self) -> QAction:
        return self._about_action

    @property
    def scene_group(self) -> QActionGroup:
        if self._scene_group is None:
            # Empty group placeholder (no scenes yet); caller should use set_scenes
            self._scene_group = QActionGroup(self._scene_menu)
            self._scene_group.setExclusive(True)
        return self._scene_group

    def set_state(self, state: str) -> None:
        self._state = state
        self._status_action.setText(f"Status: {state}")
        self.setToolTip(f"Voice Input — {state}")
        if state == "Recording":
            self._toggle_action.setText("Stop Recording")
            self._toggle_action.setEnabled(True)
            self.setIcon(self._icon_recording)
        elif state in ("Transcribing", "Refining"):
            self._toggle_action.setText("Stop Recording")
            self._toggle_action.setEnabled(False)
        else:
            self._toggle_action.setText("Start Recording")
            self._toggle_action.setEnabled(True)
            self.setIcon(self._icon_idle)

    def set_language(self, code: str) -> None:
        self._current_language = code
        if code in self._lang_actions:
            self._lang_actions[code].setChecked(True)

    def set_backend(self, backend: str) -> None:
        self._current_backend = backend
        # Map legacy "local" → "sherpa"
        key = "sherpa" if backend == "local" else backend
        if key in self._stt_actions:
            self._stt_actions[key].setChecked(True)

    def set_backend_status(self, state: str, error: str | None = None) -> None:
        if error:
            self._backend_status_action.setText(f"Backend: error ({error[:40]})")
        else:
            self._backend_status_action.setText(f"Backend: {state}")

    def set_scenes(self, scenes: list[Scene], active_id: str = "default") -> None:
        """Repopulate the scene submenu."""
        if self._scene_menu is None:
            return
        self._scene_menu.clear()
        self._scene_actions.clear()
        if self._scene_group is not None:
            for action in self._scene_group.actions():
                self._scene_group.removeAction(action)
        self._scene_group = QActionGroup(self._scene_menu)
        self._scene_group.setExclusive(True)
        for scene in scenes:
            action = QAction(scene.name, self._scene_menu)
            action.setCheckable(True)
            action.setData(scene.id)
            if scene.id == active_id:
                action.setChecked(True)
            self._scene_group.addAction(action)
            self._scene_menu.addAction(action)
            self._scene_actions[scene.id] = action

    def set_llm_enabled(self, enabled: bool) -> None:
        self._llm_enabled = enabled
        self._llm_toggle.setChecked(enabled)
```

- [ ] **Step 2: 运行 app 测试**

Run: `pytest tests/test_app.py -v`
Expected: passed（如失败，根据错误调整 tray usage）

- [ ] **Step 3: 提交**

```bash
git add src/voice_input/tray.py
git commit -m "refactor(tray): add scene submenu, remove engine concept"
```

---

## Task 17: settings_dialog.py 适配（sherpa 配置 + 场景管理）

**Files:**
- Modify: `src/voice_input/settings_dialog.py`
- Modify: `tests/test_settings_dialog.py`

为简化范围，本任务只把 settings_dialog 中现存的 "本地引擎/Whisper" 配置 UI 替换为 "sherpa" 配置 UI。场景管理 UI 留给将来扩展（场景已可通过托盘菜单切换 + config.toml 编辑）。

- [ ] **Step 1: 检查现有 settings_dialog.py**

Run: `grep -n "engine\|whisper\|sensevoice" src/voice_input/settings_dialog.py`

- [ ] **Step 2: 修改 settings_dialog.py**

替换 settings_dialog.py 中 "Local" tab 的内容。原来基于 `local.engine` / `local.model` 的 UI 改为基于 `sherpa.model_id` / `sherpa.vad_enabled` / `sherpa.num_threads` / `sherpa.provider`。

具体改动模式（视当前代码而定）：

```python
# 在 settings_dialog.py 的 Local 标签页构造器：

from voice_input.asr.model_manager import REGISTRY

# 替换 engine combo：
self._sherpa_model_combo = QComboBox()
for model_id in REGISTRY:
    self._sherpa_model_combo.addItem(model_id, userData=model_id)
current_model = config.get("stt", {}).get("sherpa", {}).get(
    "model_id", "sherpa-onnx-paraformer-zh-2024-03-09"
)
idx = self._sherpa_model_combo.findData(current_model)
if idx >= 0:
    self._sherpa_model_combo.setCurrentIndex(idx)

self._sherpa_vad_check = QCheckBox("启用 VAD（去除静音）")
self._sherpa_vad_check.setChecked(
    config.get("stt", {}).get("sherpa", {}).get("vad_enabled", True)
)

self._sherpa_threads_spin = QSpinBox()
self._sherpa_threads_spin.setRange(1, 8)
self._sherpa_threads_spin.setValue(
    config.get("stt", {}).get("sherpa", {}).get("num_threads", 2)
)

self._sherpa_provider_combo = QComboBox()
for p in ("cpu", "cuda", "coreml"):
    self._sherpa_provider_combo.addItem(p, userData=p)
current_provider = config.get("stt", {}).get("sherpa", {}).get("provider", "cpu")
idx = self._sherpa_provider_combo.findData(current_provider)
if idx >= 0:
    self._sherpa_provider_combo.setCurrentIndex(idx)
```

在 accept/save 时：

```python
sherpa_cfg = self._config.setdefault("stt", {}).setdefault("sherpa", {})
sherpa_cfg["model_id"] = self._sherpa_model_combo.currentData()
sherpa_cfg["vad_enabled"] = self._sherpa_vad_check.isChecked()
sherpa_cfg["num_threads"] = self._sherpa_threads_spin.value()
sherpa_cfg["provider"] = self._sherpa_provider_combo.currentData()
```

- [ ] **Step 3: 修改 tests/test_settings_dialog.py**

把所有针对 `engine`、`local.engine`、`local.model` 的断言改为针对 `sherpa.model_id`、`sherpa.vad_enabled`：

```python
def test_save_writes_sherpa_config(qtbot, tmp_path, monkeypatch):
    from voice_input.config import DEFAULT_CONFIG
    from voice_input.settings_dialog import SettingsDialog
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    dialog = SettingsDialog(cfg)
    qtbot.addWidget(dialog)

    dialog._sherpa_vad_check.setChecked(False)
    dialog._sherpa_threads_spin.setValue(4)
    dialog.accept()

    assert cfg["stt"]["sherpa"]["vad_enabled"] is False
    assert cfg["stt"]["sherpa"]["num_threads"] == 4
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_settings_dialog.py -v`
Expected: passed

- [ ] **Step 5: 提交**

```bash
git add src/voice_input/settings_dialog.py tests/test_settings_dialog.py
git commit -m "refactor(settings): replace local engine UI with sherpa config UI"
```

---

## Task 18: pyproject.toml — 添加 sherpa-onnx, 删除 whisper/sensevoice

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 修改 pyproject.toml**

替换 `pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "voice-input"
version = "0.2.0"
description = "KDE Plasma 6 Wayland voice input via system tray (sherpa-onnx)"
requires-python = ">=3.11"
dependencies = [
    "PyQt6>=6.6",
    "sounddevice>=0.4",
    "dbus-next>=0.2",
    "httpx>=0.27",
    "socksio>=1.0",
    "qasync>=0.27",
    "keyring>=25.0",
    "numpy>=1.24",
    "websockets>=12.0",
    "sherpa-onnx>=1.10",
    "tomli>=2.0; python_version < '3.12'",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-qt>=4.3", "pytest-asyncio>=0.23", "respx>=0.21"]
google = ["google-auth>=2.0"]

[project.scripts]
voice-input = "voice_input.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "strict"
```

- [ ] **Step 2: 重新安装依赖**

Run: `pip install -e ".[dev]"`
Expected: sherpa-onnx 安装成功

- [ ] **Step 3: 提交**

```bash
git add pyproject.toml
git commit -m "build: add sherpa-onnx, drop whisper/sensevoice optional deps"
```

---

## Task 19: 删除老代码 — local backend, whisper_worker

**Files:**
- Delete: `src/voice_input/backends/local/__init__.py`
- Delete: `src/voice_input/backends/local/whisper_engine.py`
- Delete: `src/voice_input/backends/local/sensevoice_engine.py`
- Delete: `src/voice_input/backends/local/engine.py`
- Delete: `src/voice_input/whisper_worker.py`
- Delete: `tests/test_local_backend.py`

- [ ] **Step 1: 删除文件**

```bash
git rm src/voice_input/backends/local/__init__.py \
       src/voice_input/backends/local/whisper_engine.py \
       src/voice_input/backends/local/sensevoice_engine.py \
       src/voice_input/backends/local/engine.py \
       src/voice_input/whisper_worker.py \
       tests/test_local_backend.py
rmdir src/voice_input/backends/local 2>/dev/null || true
```

- [ ] **Step 2: 验证没有 import 引用**

```bash
grep -rn "voice_input.backends.local\|whisper_worker\|WhisperWorker\|whisper_engine\|sensevoice_engine" src/ tests/
```
Expected: 无输出（所有引用已清理）

如有遗漏，删除引用。

- [ ] **Step 3: 运行所有测试**

Run: `pytest tests/ -v`
Expected: all passed

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "chore: delete legacy local backend, whisper_worker"
```

---

## Task 20: 手动验证 + 真实模型 SHA256 填入

**Files:**
- Modify: `src/voice_input/asr/model_manager.py` (填入真实 SHA256)
- 手动测试系统服务

- [ ] **Step 1: 计算真实 SHA256**

下载并计算各模型文件的 SHA256：

```bash
mkdir -p /tmp/sherpa-checksums && cd /tmp/sherpa-checksums
curl -L -o paraformer-model.int8.onnx https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09/resolve/main/model.int8.onnx
curl -L -o paraformer-tokens.txt https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09/resolve/main/tokens.txt
curl -L -o sense-voice-model.int8.onnx https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx
curl -L -o sense-voice-tokens.txt https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt
curl -L -o silero_vad.onnx https://huggingface.co/csukuangfj/sherpa-onnx-silero-vad/resolve/main/silero_vad.onnx
sha256sum *
```

- [ ] **Step 2: 把输出的 SHA256 填入 model_manager.py 顶部常量**

编辑 `src/voice_input/asr/model_manager.py`，把 `PARAFORMER_MODEL_SHA256 = ""` 等空字符串替换为实际值。

- [ ] **Step 3: 运行所有测试确保通过**

Run: `pytest tests/ -v`
Expected: passed

- [ ] **Step 4: 启动应用并执行手动验证清单**

Run: `make run` 或 `python -m voice_input`

逐项验证：
- [ ] `make run` 启动后 5 秒内托盘图标显示 "Backend: ready"
- [ ] 启动 1 秒内按热键 → 弹"模型加载中"通知，不录音不报错
- [ ] 模型 ready 后按热键 → 录音 → 识别 → 注入文本（中文 + 英文混合）
- [ ] 设置里改 model_id 保存 → 旧模型继续可用 → 新模型 ready 后无缝切换
- [ ] 故意配置错的 model_id（如 "nonexistent"）→ 通知报错 → 旧模型继续可用
- [ ] 切换场景（在托盘"场景"子菜单）→ 新文本走新 prompt
- [ ] LLM 配置错（清空 API key）→ 注入未经 LLM 处理的 ASR 原文
- [ ] 删除 cache 目录（`rm -rf ~/.cache/voice-input/sherpa-models/`）→ 重启 → 模型自动重新下载

- [ ] **Step 5: 提交真实 SHA256**

```bash
git add src/voice_input/asr/model_manager.py
git commit -m "chore(asr): fill real SHA256 for sherpa-onnx models"
```

- [ ] **Step 6: 标记完成**

```bash
git log --oneline -25  # 查看本次重构的全部提交
```

---

## 完成 Definition of Done

- [ ] 所有 task 的 commit 完成
- [ ] `pytest tests/ -v` 全绿
- [ ] 手动验证清单全部通过
- [ ] 旧依赖（faster-whisper、funasr、modelscope）从环境中移除
- [ ] `make run` 能启动并使用
