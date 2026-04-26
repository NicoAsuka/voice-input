# tests/test_config.py
import tomllib
from pathlib import Path

from voice_input.config import AppConfig, load_config, save_config, DEFAULT_CONFIG


def test_default_config_has_required_sections():
    cfg = DEFAULT_CONFIG
    assert cfg["hotkey"]["mode"] == "toggle"
    assert cfg["hotkey"]["key"] == "Meta+Space"
    assert cfg["stt"]["backend"] == "local"
    assert cfg["stt"]["local"]["engine"] == "whisper"
    assert cfg["stt"]["local"]["model"] == "medium"
    assert cfg["stt"]["local"]["language"] == "zh"
    assert cfg["stt"]["local"]["device"] == "auto"
    assert cfg["llm"]["enabled"] is True
    assert cfg["llm"]["api_base"] == "https://api.openai.com/v1"
    assert cfg["llm"]["model"] == "gpt-4o-mini"
    assert cfg["audio"]["device"] == "default"
    assert cfg["audio"]["sample_rate"] == 16000
    assert cfg["ui"]["overlay_margin_bottom"] == 80


def test_default_config_has_stt_section():
    cfg = DEFAULT_CONFIG
    assert cfg["stt"]["backend"] == "local"
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
    cfg["stt"]["local"]["language"] = "en"
    save_config(cfg, config_dir=config_dir)
    reloaded = load_config(config_dir=config_dir)
    assert reloaded["stt"]["local"]["language"] == "en"


def test_load_config_merges_partial_file(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    # Write a partial config — only hotkey section
    (config_dir / "config.toml").write_text('[hotkey]\nmode = "hold"\n')
    cfg = load_config(config_dir=config_dir)
    # Overridden value
    assert cfg["hotkey"]["mode"] == "hold"
    # Default values still present
    assert cfg["stt"]["local"]["model"] == "medium"
    assert cfg["llm"]["enabled"] is True


def test_load_config_migrates_legacy_whisper_section(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        "\n".join(
            [
                "[whisper]",
                'model = "small"',
                'language = "en"',
                'device = "cpu"',
                "",
            ]
        )
    )

    cfg = load_config(config_dir=config_dir)

    assert cfg["stt"]["local"]["model"] == "small"
    assert cfg["stt"]["local"]["language"] == "en"
    assert cfg["stt"]["local"]["device"] == "cpu"


def test_default_config_has_stt_local():
    assert "stt" in DEFAULT_CONFIG
    assert "local" in DEFAULT_CONFIG["stt"]
    local = DEFAULT_CONFIG["stt"]["local"]
    assert local["engine"] == "whisper"
    assert local["model"] == "medium"
    assert local["language"] == "zh"
    assert local["device"] == "auto"


def test_default_config_no_whisper_section():
    assert "whisper" not in DEFAULT_CONFIG


def test_xdg_paths():
    from voice_input.config import xdg_config_dir, xdg_cache_dir, xdg_data_dir
    assert xdg_config_dir().name == "voice-input"
    assert xdg_cache_dir().name == "voice-input"
    assert xdg_data_dir().name == "voice-input"
