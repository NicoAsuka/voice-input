# src/voice_input/injector.py
from __future__ import annotations

import enum
import logging
import shutil
import subprocess
import time

log = logging.getLogger(__name__)


class InjectionMethod(enum.Enum):
    WTYPE = "wtype"
    YDOTOOL = "ydotool"
    CLIPBOARD = "clipboard"
    NONE = "none"


class TextInjector:
    """Injects text into the focused Wayland window.

    Priority: wtype → ydotool → clipboard (wl-copy + Ctrl+V).
    Handles fcitx5/ibus IM state save/restore and clipboard protection.
    """

    def __init__(self) -> None:
        self.method = self._detect_method()
        log.info("Text injection method: %s", self.method.value)

    @staticmethod
    def _detect_method() -> InjectionMethod:
        if shutil.which("wtype"):
            return InjectionMethod.WTYPE
        if shutil.which("ydotool"):
            return InjectionMethod.YDOTOOL
        if shutil.which("wl-copy") and shutil.which("wl-paste"):
            return InjectionMethod.CLIPBOARD
        return InjectionMethod.NONE

    def inject(self, text: str) -> bool:
        """Inject text into the focused application. Returns True on success."""
        if not text or self.method == InjectionMethod.NONE:
            log.warning("No injection method available or empty text")
            return False

        im_state = self._save_im_state()
        if im_state == "active":
            self._deactivate_im()

        clipboard_backup = None
        if self.method == InjectionMethod.CLIPBOARD:
            clipboard_backup = self._save_clipboard()

        try:
            success = self._do_inject(text)
        finally:
            if clipboard_backup is not None:
                time.sleep(0.1)
                self._restore_clipboard(clipboard_backup)
            if im_state == "active":
                self._activate_im()

        return success

    def _do_inject(self, text: str) -> bool:
        if self.method == InjectionMethod.WTYPE:
            cmd = self._build_inject_command(text)
            return self._run(cmd)
        elif self.method == InjectionMethod.YDOTOOL:
            cmd = ["ydotool", "type", "--", text]
            return self._run(cmd)
        elif self.method == InjectionMethod.CLIPBOARD:
            copy_cmd, paste_cmd = self._build_clipboard_commands(text)
            if not self._run(copy_cmd):
                return False
            time.sleep(0.05)
            return self._run(paste_cmd)
        return False

    def _build_inject_command(self, text: str) -> list[str]:
        return ["wtype", "--", text]

    def _build_clipboard_commands(self, text: str) -> tuple[list[str], list[str]]:
        copy_cmd = ["wl-copy", "--", text]
        if shutil.which("wtype"):
            paste_cmd = ["wtype", "-M", "ctrl", "v", "-m", "ctrl"]
        elif shutil.which("ydotool"):
            paste_cmd = ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]
        else:
            paste_cmd = ["wtype", "-M", "ctrl", "v", "-m", "ctrl"]  # best effort
        return copy_cmd, paste_cmd

    def _save_im_state(self) -> str:
        """Detect and save current input method state. Returns 'active', 'inactive', or 'unknown'."""
        if shutil.which("fcitx5-remote"):
            try:
                result = subprocess.run(
                    ["fcitx5-remote"],
                    capture_output=True, text=True, timeout=2,
                )
                code = result.stdout.strip()
                if code == "2":
                    return "active"
                elif code == "1":
                    return "inactive"
            except Exception as e:
                log.debug("fcitx5-remote query failed: %s", e)

        if shutil.which("ibus"):
            try:
                result = subprocess.run(
                    ["ibus", "read-cache"],
                    capture_output=True, text=True, timeout=2,
                )
            except Exception:
                pass

        return "unknown"

    def _deactivate_im(self) -> None:
        if shutil.which("fcitx5-remote"):
            self._run(["fcitx5-remote", "-c"])
        elif shutil.which("ibus"):
            self._run(["ibus", "engine", "xkb:us::eng"])

    def _activate_im(self) -> None:
        if shutil.which("fcitx5-remote"):
            self._run(["fcitx5-remote", "-o"])

    def _save_clipboard(self) -> str | None:
        if not shutil.which("wl-paste"):
            return None
        try:
            result = subprocess.run(
                ["wl-paste", "--no-newline"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            log.debug("wl-paste failed: %s", e)
        return None

    def _restore_clipboard(self, content: str) -> None:
        if content and shutil.which("wl-copy"):
            self._run(["wl-copy", "--", content])

    @staticmethod
    def _run(cmd: list[str]) -> bool:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                log.warning("Command failed: %s stderr=%s", cmd, result.stderr)
                return False
            return True
        except Exception as e:
            log.error("Command error: %s %s", cmd, e)
            return False
