# tests/test_config.py
import tomllib
from pathlib import Path

from voice_input.config import (
    AppConfig,
    load_config,
    save_config,
    DEFAULT_CONFIG,
    _migrate_legacy_local_config,
)


def test_default_config_has_required_sections():
    cfg = DEFAULT_CONFIG
    assert cfg["hotkey"]["mode"] == "toggle"
    assert cfg["hotkey"]["key"] == "Meta+Space"
    assert cfg["stt"]["backend"] == "sherpa"
    assert cfg["stt"]["sherpa"]["model_id"] == "sherpa-onnx-paraformer-zh-2024-03-09"
    assert cfg["stt"]["sherpa"]["vad_enabled"] is True
    assert cfg["stt"]["language"] == "zh"
    assert cfg["llm"]["enabled"] is True
    assert cfg["llm"]["api_base"] == "https://api.openai.com/v1"
    assert cfg["llm"]["model"] == "gpt-4o-mini"
    assert cfg["audio"]["device"] == "default"
    assert cfg["audio"]["sample_rate"] == 16000
    assert cfg["ui"]["overlay_margin_bottom"] == 80


def test_default_config_has_stt_section():
    cfg = DEFAULT_CONFIG
    assert cfg["stt"]["backend"] == "sherpa"
    assert cfg["stt"]["openai"]["api_base"] == "https://api.openai.com/v1"
    assert cfg["stt"]["openai"]["model"] == "whisper-1"
    assert cfg["stt"]["google"]["credentials_path"] == ""
    assert cfg["stt"]["volcengine"]["app_id"] == ""


def test_load_config_creates_default_when_missing(tmp_path):
    config_dir = tmp_path / "config"
    cfg = load_config(config_dir=config_dir)
    assert cfg["hotkey"]["mode"] == "toggle"
    # File should have been created
    assert (config_dir / "config.toml").exists()


def test_save_and_reload(tmp_path):
    config_dir = tmp_path / "config"
    cfg = load_config(config_dir=config_dir)
    cfg["stt"]["language"] = "en"
    save_config(cfg, config_dir=config_dir)
    reloaded = load_config(config_dir=config_dir)
    assert reloaded["stt"]["language"] == "en"


def test_load_config_merges_partial_file(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    # Write a partial config -- only hotkey section
    (config_dir / "config.toml").write_text('[hotkey]\nmode = "hold"\n')
    cfg = load_config(config_dir=config_dir)
    # Overridden value
    assert cfg["hotkey"]["mode"] == "hold"
    # Default values still present
    assert cfg["stt"]["sherpa"]["model_id"] == "sherpa-onnx-paraformer-zh-2024-03-09"
    assert cfg["llm"]["enabled"] is True


def test_load_config_migrates_legacy_local_to_sherpa(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        "\n".join(
            [
                "[stt]",
                'backend = "local"',
                'language = "en"',
                "",
                "[stt.local]",
                'language = "en"',
                'device = "cpu"',
                "",
            ]
        )
    )

    cfg = load_config(config_dir=config_dir)

    assert cfg["stt"]["backend"] == "sherpa"
    assert cfg["stt"]["language"] == "en"


def test_default_config_has_sherpa_section():
    assert "stt" in DEFAULT_CONFIG
    assert "sherpa" in DEFAULT_CONFIG["stt"]
    sherpa = DEFAULT_CONFIG["stt"]["sherpa"]
    assert sherpa["model_id"] == "sherpa-onnx-paraformer-zh-2024-03-09"
    assert sherpa["vad_enabled"] is True
    assert sherpa["num_threads"] == 2
    assert sherpa["provider"] == "cpu"


def test_default_config_backend_is_sherpa():
    assert DEFAULT_CONFIG["stt"]["backend"] == "sherpa"


def test_default_config_has_postprocess_section():
    assert "postprocess" in DEFAULT_CONFIG
    pp = DEFAULT_CONFIG["postprocess"]
    assert pp["enabled"] is True
    assert pp["active_scene"] == "default"
    assert isinstance(pp["scenes"], list)
    assert len(pp["scenes"]) >= 4
    ids = [s["id"] for s in pp["scenes"]]
    assert "default" in ids
    assert "code" in ids


def test_default_config_no_local_section():
    assert "stt" in DEFAULT_CONFIG
    assert "local" not in DEFAULT_CONFIG["stt"]


def test_migrate_legacy_local_config_remaps_backend():
    cfg = {"stt": {"backend": "sherpa", "language": "zh"}}
    user_cfg = {"stt": {"backend": "local"}}
    result = _migrate_legacy_local_config(cfg, user_cfg)
    assert result["stt"]["backend"] == "sherpa"


def test_migrate_legacy_local_config_preserves_language():
    # When cfg already has stt.language set, setdefault won't override it
    cfg = {"stt": {"backend": "sherpa", "language": "zh"}}
    user_cfg = {"stt": {"backend": "local", "local": {"language": "en"}}}
    result = _migrate_legacy_local_config(cfg, user_cfg)
    assert result["stt"]["language"] == "zh"  # setdefault preserves existing
    assert result["stt"]["backend"] == "sherpa"


def test_migrate_legacy_local_config_sets_language_when_absent():
    # When cfg does NOT have stt.language, migration sets it from local
    cfg = {"stt": {"backend": "sherpa"}}
    user_cfg = {"stt": {"backend": "local", "local": {"language": "en"}}}
    result = _migrate_legacy_local_config(cfg, user_cfg)
    assert result["stt"]["language"] == "en"
    assert result["stt"]["backend"] == "sherpa"


def test_default_config_no_whisper_section():
    assert "whisper" not in DEFAULT_CONFIG


def test_xdg_paths():
    from voice_input.config import xdg_config_dir, xdg_cache_dir, xdg_data_dir
    assert xdg_config_dir().name == "voice-input"
    assert xdg_cache_dir().name == "voice-input"
    assert xdg_data_dir().name == "voice-input"
