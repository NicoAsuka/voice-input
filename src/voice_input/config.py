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
        "backend": "sherpa",
        "language": "zh",
        "sherpa": {
            "model_id": "sherpa-onnx-paraformer-zh-2024-03-09",
            "vad_enabled": True,
            "num_threads": 2,
            "provider": "cpu",
        },
        "openai": {
            "api_base": "https://api.openai.com/v1",
            "model": "whisper-1",
        },
        "google": {
            "credentials_path": "",
        },
        "volcengine": {
            "app_id": "",
            "resource_id": "volc.seedasr.sauc.duration",
        },
    },
    "llm": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "postprocess": {
        "enabled": True,
        "active_scene": "default",
        "scenes": [
            {
                "id": "default",
                "name": "默认",
                "prompt": "修正 ASR 识别错误，保持原意，返回纯文本。",
            },
            {
                "id": "code",
                "name": "代码场景",
                "prompt": (
                    "这是程序员说的话。修正中文同音字错误，把英文技术术语恢复"
                    "（例如：'派森' → 'Python'，'瑞克特' → 'React'）。返回纯文本。"
                ),
            },
            {
                "id": "translate-en",
                "name": "翻译为英文",
                "prompt": "把以下中文翻译为地道英文，只返回译文。",
            },
            {
                "id": "polish",
                "name": "口语转书面",
                "prompt": "把口语化的中文改为书面表达，删除语气词和重复，保持原意。",
            },
        ],
    },
    "audio": {
        "device": "default",
        "sample_rate": 16000,
        "silence_threshold": 0.01,
        "silence_timeout_ms": 2000,
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
        if "\n" in v or '"' in v:
            # TOML multiline basic string: escape \ and " (including """ sequences)
            chars = list(v)
            parts: list[str] = []
            i = 0
            while i < len(chars):
                if chars[i] == "\\":
                    parts.append("\\\\")
                    i += 1
                elif chars[i : i + 3] == ['"', '"', '"']:
                    parts.append('\\"\\"\\"')
                    i += 3
                elif chars[i] == '"':
                    parts.append('\\"')
                    i += 1
                else:
                    parts.append(chars[i])
                    i += 1
            return '"""' + "".join(parts) + '"""'
        if "\\" in v:
            # TOML basic string: backslash must be escaped
            return '"' + v.replace("\\", "\\\\") + '"'
        return f'"{v}"'
    return str(v)


def _to_toml(data: dict, prefix: str = "") -> str:
    """Simple dict-to-TOML serializer with [[arrays of tables]]."""
    lines: list[str] = []
    scalars: list[tuple[str, object]] = []
    sections: list[tuple[str, dict]] = []
    array_tables: list[tuple[str, list]] = []

    for k, v in data.items():
        if isinstance(v, dict):
            sections.append((k, v))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            array_tables.append((k, v))
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
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                array_full = f"{full}.{k}"
                for entry in v:
                    lines.append(f"\n[[{array_full}]]")
                    for ek, ev in entry.items():
                        lines.append(f"{ek} = {_toml_value(ev)}")
            else:
                lines.append(f"{k} = {_toml_value(v)}")

    for k, v in array_tables:
        full = f"{prefix}.{k}" if prefix else k
        for entry in v:
            lines.append(f"\n[[{full}]]")
            for ek, ev in entry.items():
                lines.append(f"{ek} = {_toml_value(ev)}")

    return "\n".join(lines)


AppConfig = dict  # type alias for clarity


def _migrate_legacy_local_config(cfg: dict, user_cfg: dict) -> dict:
    """Map old config keys to current schema."""
    user_stt = user_cfg.get("stt", {})
    # Old "local" backend -> "sherpa"
    if user_stt.get("backend") == "local":
        cfg.setdefault("stt", {})["backend"] = "sherpa"
    # Old [stt.local] language -> top-level stt.language
    legacy_local = user_stt.get("local", {})
    if isinstance(legacy_local, dict) and "language" in legacy_local:
        cfg.setdefault("stt", {}).setdefault("language", legacy_local["language"])
    # Old [whisper] section -> ignore (whisper backend removed)

    # Old llm.enabled -> postprocess.enabled
    user_llm = user_cfg.get("llm", {})
    if isinstance(user_llm, dict) and "enabled" in user_llm:
        user_pp = user_cfg.get("postprocess", {})
        if not isinstance(user_pp, dict) or "enabled" not in user_pp:
            cfg.setdefault("postprocess", {})["enabled"] = user_llm["enabled"]

    return cfg


def load_config(config_dir: Path | None = None) -> AppConfig:
    config_dir = config_dir or xdg_config_dir()
    config_file = config_dir / "config.toml"
    if config_file.exists():
        with open(config_file, "rb") as f:
            user_cfg = tomllib.load(f)
        cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)
        cfg = _migrate_legacy_local_config(cfg, user_cfg)
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
