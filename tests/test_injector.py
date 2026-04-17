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


def test_build_wtype_command():
    injector = TextInjector.__new__(TextInjector)
    injector.method = InjectionMethod.WTYPE
    cmd = injector._build_inject_command("hello world")
    assert cmd == ["wtype", "--", "hello world"]


def test_build_clipboard_paste_command():
    injector = TextInjector.__new__(TextInjector)
    injector.method = InjectionMethod.CLIPBOARD
    # For clipboard mode, we copy then simulate Ctrl+V
    copy_cmd, paste_cmd = injector._build_clipboard_commands("hello")
    assert copy_cmd == ["wl-copy", "--", "hello"]
    assert paste_cmd[0] == "wtype"  # wtype -M ctrl v


def test_fcitx5_save_restore():
    """fcitx5 state should be saved and restorable."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="2\n"  # 2 = active
        )
        injector = TextInjector.__new__(TextInjector)
        injector.method = InjectionMethod.WTYPE
        state = injector._save_im_state()
        assert state == "active"
