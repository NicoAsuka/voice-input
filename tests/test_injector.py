# tests/test_injector.py
import subprocess
from unittest.mock import patch, MagicMock

from voice_input.injector import TextInjector, InjectionMethod


def test_detect_clipboard_paste_method():
    """When wl-copy and ydotool are available, method is CLIPBOARD_PASTE."""
    def which_side_effect(name):
        if name in ("wl-copy", "wl-paste", "ydotool"):
            return f"/usr/bin/{name}"
        return None

    with patch("shutil.which", side_effect=which_side_effect):
        injector = TextInjector()
        assert injector.method == InjectionMethod.CLIPBOARD_PASTE
        assert injector.is_ready() is True


def test_detect_none_when_missing():
    """When tools are missing, method is NONE."""
    with patch("shutil.which", return_value=None):
        injector = TextInjector()
        assert injector.method == InjectionMethod.NONE
        assert injector.is_ready() is False


def test_inject_sets_last_error_on_missing_tools():
    """inject() sets last_error when required tools are missing."""
    with patch("shutil.which", return_value=None):
        injector = TextInjector()
        result = injector.inject("hello")
        assert result is False
        assert "missing tools" in injector.last_error


def test_inject_sets_last_error_on_empty_text():
    injector = TextInjector.__new__(TextInjector)
    injector._has_wl_copy = True
    injector._has_wl_paste = True
    injector._has_ydotool = True
    injector._paste_keys = []
    injector.method = InjectionMethod.CLIPBOARD_PASTE
    injector.last_error = ""
    result = injector.inject("")
    assert result is False
    assert "empty" in injector.last_error
