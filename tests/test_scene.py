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
