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
