from __future__ import annotations

from dataclasses import dataclass

from voice_input.postprocess.llm import SYSTEM_PROMPT


@dataclass(frozen=True)
class Scene:
    id: str
    name: str
    prompt: str


DEFAULT_SCENE = Scene(
    id="default",
    name="默认",
    prompt=SYSTEM_PROMPT,
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
