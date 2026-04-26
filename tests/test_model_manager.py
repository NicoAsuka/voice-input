from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest
import respx
from httpx import Response

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
