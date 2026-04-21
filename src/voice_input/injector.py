# src/voice_input/injector.py
from __future__ import annotations

import enum
import logging
import shutil
import subprocess
import time

log = logging.getLogger(__name__)

# ydotool key codes
KEY_LEFTCTRL = 29
KEY_LEFTSHIFT = 42
KEY_V = 47

# Paste key sequences: {method_name: key_sequence}
PASTE_METHODS = {
    "ctrl_v": [f"{KEY_LEFTCTRL}:1", f"{KEY_V}:1", f"{KEY_V}:0", f"{KEY_LEFTCTRL}:0"],
    "ctrl_shift_v": [
        f"{KEY_LEFTCTRL}:1", f"{KEY_LEFTSHIFT}:1",
        f"{KEY_V}:1", f"{KEY_V}:0",
        f"{KEY_LEFTSHIFT}:0", f"{KEY_LEFTCTRL}:0",
    ],
}


class InjectionMethod(enum.Enum):
    WTYPE = "wtype"
    YDOTOOL = "ydotool"
    CLIPBOARD = "clipboard"
    NONE = "none"


class TextInjector:
    """Injects text into the focused Wayland window.

    Uses clipboard (wl-copy) + ydotool paste shortcut.
    wtype text-typing is broken on KDE Plasma 6 (virtual keyboard protocol
    not supported), so we always use the clipboard route.
    """

    def __init__(self, paste_method: str = "ctrl_v") -> None:
        self.method = self._detect_method()
        self._has_wl_copy = bool(shutil.which("wl-copy"))
        self._has_wl_paste = bool(shutil.which("wl-paste"))
        self._has_ydotool = bool(shutil.which("ydotool"))
        self._paste_keys = PASTE_METHODS.get(paste_method, PASTE_METHODS["ctrl_v"])
        if self._has_wl_copy and self._has_ydotool:
            log.info("Text injection: wl-copy + ydotool (paste method: %s)", paste_method)
        else:
            log.warning(
                "Text injection may not work: wl-copy=%s ydotool=%s",
                self._has_wl_copy, self._has_ydotool,
            )

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
        """Inject text into the focused application via clipboard paste."""
        if not text:
            return False
        if not self._has_wl_copy or not self._has_ydotool:
            log.error("wl-copy or ydotool not available, cannot inject text")
            return False

        # Save current clipboard
        old_clip = self._get_clipboard()

        # Copy text to clipboard
        if not self._run(["wl-copy", "--", text]):
            return False

        time.sleep(0.05)

        # Paste via ydotool
        ok = self._run(["ydotool", "key"] + self._paste_keys)

        # Restore clipboard after a short delay
        if old_clip is not None:
            time.sleep(0.15)
            self._run(["wl-copy", "--", old_clip])

        if ok:
            log.info("Text injected (%d chars)", len(text))
        return ok

    def _build_inject_command(self, text: str) -> list[str]:
        return ["wtype", "--", text]

    def _build_clipboard_commands(self, text: str) -> tuple[list[str], list[str]]:
        return ["wl-copy", "--", text], ["wtype", "-M", "ctrl", "v", "-m", "ctrl"]

    def _save_im_state(self) -> str:
        """Detect current input method state."""
        if shutil.which("fcitx5-remote"):
            try:
                result = subprocess.run(
                    ["fcitx5-remote"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                code = result.stdout.strip()
                if code == "2":
                    return "active"
                if code == "1":
                    return "inactive"
            except Exception as e:
                log.debug("fcitx5-remote query failed: %s", e)
        return "unknown"

    def _get_clipboard(self) -> str | None:
        if not self._has_wl_paste:
            return None
        try:
            r = subprocess.run(
                ["wl-paste", "--no-newline"],
                capture_output=True, text=True, timeout=2,
            )
            return r.stdout if r.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _run(cmd: list[str]) -> bool:
        try:
            if cmd[0].endswith("wl-copy"):
                # wl-copy forks a background process to serve the clipboard.
                # capture_output=True pipes stdout/stderr, which prevents the
                # forked parent from closing those fds and causes subprocess.run
                # to block until timeout.  Use Popen without pipes instead.
                p = subprocess.Popen(cmd)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                    return False
                return p.returncode == 0
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode != 0:
                log.warning("Command failed: %s stderr=%s", cmd, r.stderr.strip())
                return False
            return True
        except Exception as e:
            log.error("Command error: %s %s", cmd, e)
            return False
