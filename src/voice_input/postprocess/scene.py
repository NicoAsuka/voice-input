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
