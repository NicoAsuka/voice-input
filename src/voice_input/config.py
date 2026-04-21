# src/voice_input/config.py
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

if sys.version_info >= (3, 12):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

APP_NAME = "voice-input"

DEFAULT_CONFIG: dict = {
    "hotkey": {
        "mode": "toggle",
        "key": "Meta+Space",
    },
    "stt": {
        "backend": "local",
        "local": {
            "engine": "whisper",
            "model": "medium",
            "language": "zh",
            "device": "auto",
        },
    },
    "llm": {
        "enabled": True,
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "audio": {
        "device": "default",
        "sample_rate": 16000,
    },
    "ui": {
        "overlay_margin_bottom": 80,
    },
    "inject": {
        "paste_method": "ctrl_v",
    },
}


def xdg_config_dir() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_NAME


def xdg_cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / APP_NAME


def xdg_data_dir() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_NAME


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Returns new dict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(v)
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def _to_toml(data: dict, prefix: str = "") -> str:
    """Simple dict-to-TOML serializer (flat sections only)."""
    lines: list[str] = []
    scalars: list[tuple[str, object]] = []
    sections: list[tuple[str, dict]] = []
    for k, v in data.items():
        if isinstance(v, dict):
            sections.append((k, v))
        else:
            scalars.append((k, v))
    for k, v in scalars:
        lines.append(f"{k} = {_toml_value(v)}")
    for section_name, section_dict in sections:
        full = f"{prefix}.{section_name}" if prefix else section_name
        lines.append(f"\n[{full}]")
        for k, v in section_dict.items():
            if isinstance(v, dict):
                lines.append(_to_toml({k: v}, prefix=full))
            else:
                lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines)


AppConfig = dict  # type alias for clarity


def load_config(config_dir: Path | None = None) -> AppConfig:
    config_dir = config_dir or xdg_config_dir()
    config_file = config_dir / "config.toml"
    if config_file.exists():
        with open(config_file, "rb") as f:
            user_cfg = tomllib.load(f)
        cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)
    else:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        # Write defaults so the user has a template
        save_config(cfg, config_dir=config_dir)
    return cfg


def save_config(cfg: AppConfig, config_dir: Path | None = None) -> None:
    config_dir = config_dir or xdg_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(_to_toml(cfg) + "\n")
