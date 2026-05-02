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
