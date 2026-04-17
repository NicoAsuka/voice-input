# tests/test_hotkey.py
from unittest.mock import AsyncMock, patch, MagicMock

from voice_input.hotkey import HotkeyManager, parse_key_string


def test_parse_key_string_meta_space():
    modifiers, key = parse_key_string("Meta+Space")
    assert "Meta" in modifiers
    assert key == "Space"


def test_parse_key_string_single_key():
    modifiers, key = parse_key_string("F9")
    assert modifiers == []
    assert key == "F9"


def test_parse_key_string_multi_modifier():
    modifiers, key = parse_key_string("Ctrl+Shift+A")
    assert "Ctrl" in modifiers
    assert "Shift" in modifiers
    assert key == "A"


def test_hotkey_manager_init_toggle():
    mgr = HotkeyManager(mode="toggle", key="Meta+Space")
    assert mgr.mode == "toggle"
    assert mgr.key_string == "Meta+Space"


def test_hotkey_manager_init_hold():
    mgr = HotkeyManager(mode="hold", key="Meta+Space")
    assert mgr.mode == "hold"
