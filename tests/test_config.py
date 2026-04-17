# tests/test_config.py
import tomllib
from pathlib import Path

from voice_input.config import AppConfig, load_config, save_config, DEFAULT_CONFIG


def test_default_config_has_required_sections():
    cfg = DEFAULT_CONFIG
    assert cfg["hotkey"]["mode"] == "toggle"
    assert cfg["hotkey"]["key"] == "Meta+Space"
    assert cfg["whisper"]["model"] == "medium"
    assert cfg["whisper"]["language"] == "zh"
    assert cfg["whisper"]["device"] == "auto"
    assert cfg["llm"]["enabled"] is True
    assert cfg["llm"]["api_base"] == "https://api.openai.com/v1"
    assert cfg["llm"]["model"] == "gpt-4o-mini"
    assert cfg["audio"]["device"] == "default"
    assert cfg["audio"]["sample_rate"] == 16000
    assert cfg["ui"]["overlay_margin_bottom"] == 80


def test_load_config_creates_default_when_missing(tmp_path):
    config_dir = tmp_path / "config"
    cfg = load_config(config_dir=config_dir)
    assert cfg["hotkey"]["mode"] == "toggle"
    # File should have been created
    assert (config_dir / "config.toml").exists()


def test_save_and_reload(tmp_path):
    config_dir = tmp_path / "config"
    cfg = load_config(config_dir=config_dir)
    cfg["whisper"]["language"] = "en"
    save_config(cfg, config_dir=config_dir)
    reloaded = load_config(config_dir=config_dir)
    assert reloaded["whisper"]["language"] == "en"


def test_load_config_merges_partial_file(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    # Write a partial config — only hotkey section
    (config_dir / "config.toml").write_text('[hotkey]\nmode = "hold"\n')
    cfg = load_config(config_dir=config_dir)
    # Overridden value
    assert cfg["hotkey"]["mode"] == "hold"
    # Default values still present
    assert cfg["whisper"]["model"] == "medium"
    assert cfg["llm"]["enabled"] is True


def test_xdg_paths():
    from voice_input.config import xdg_config_dir, xdg_cache_dir, xdg_data_dir
    assert xdg_config_dir().name == "voice-input"
    assert xdg_cache_dir().name == "voice-input"
    assert xdg_data_dir().name == "voice-input"
