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
