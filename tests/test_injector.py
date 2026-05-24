# tests/test_injector.py
import subprocess
from unittest.mock import patch, MagicMock

from voice_input.injector import TextInjector, InjectionMethod


def test_detect_wtype(tmp_path):
    with patch("shutil.which", return_value="/usr/bin/wtype"):
        injector = TextInjector()
        assert injector.method == InjectionMethod.WTYPE


def test_detect_ydotool_fallback():
    def which_side_effect(name):
        if name == "wtype":
            return None
        if name == "ydotool":
            return "/usr/bin/ydotool"
        return None

    with patch("shutil.which", side_effect=which_side_effect):
        injector = TextInjector()
        assert injector.method == InjectionMethod.YDOTOOL


def test_detect_clipboard_fallback():
    def which_side_effect(name):
        if name in ("wl-copy", "wl-paste"):
            return f"/usr/bin/{name}"
        return None

    with patch("shutil.which", side_effect=which_side_effect):
        injector = TextInjector()
        assert injector.method == InjectionMethod.CLIPBOARD
