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
    """Actual injection execution path used by inject()."""
    CLIPBOARD_PASTE = "clipboard_paste"  # wl-copy + ydotool key
    NONE = "none"


class TextInjector:
    """Injects text into the focused Wayland window.

    Uses clipboard (wl-copy) + ydotool paste shortcut.
    wtype text-typing is broken on KDE Plasma 6 (virtual keyboard protocol
    not supported), so we always use the clipboard route.
    """

    def __init__(self, paste_method: str = "ctrl_v") -> None:
        self._has_wl_copy = bool(shutil.which("wl-copy"))
        self._has_wl_paste = bool(shutil.which("wl-paste"))
        self._has_ydotool = bool(shutil.which("ydotool"))
        self._paste_keys = PASTE_METHODS.get(paste_method, PASTE_METHODS["ctrl_v"])
        self.last_error: str = ""

        if self._has_wl_copy and self._has_ydotool:
            self.method = InjectionMethod.CLIPBOARD_PASTE
            log.info("Text injection: wl-copy + ydotool (paste method: %s)", paste_method)
        else:
            self.method = InjectionMethod.NONE
            log.warning(
                "Text injection unavailable: wl-copy=%s ydotool=%s",
                self._has_wl_copy, self._has_ydotool,
            )

    def is_ready(self) -> bool:
        """Return True if injection can work (all required tools available)."""
        return self.method == InjectionMethod.CLIPBOARD_PASTE

    def inject(self, text: str) -> bool:
        """Inject text into the focused application via clipboard paste.

        On failure, sets self.last_error with a human-readable reason.
        """
        self.last_error = ""
        if not text:
            self.last_error = "empty text"
            return False
        if not self._has_wl_copy or not self._has_ydotool:
            missing = []
            if not self._has_wl_copy:
                missing.append("wl-copy")
            if not self._has_ydotool:
                missing.append("ydotool")
            self.last_error = f"missing tools: {', '.join(missing)}"
            log.error("Cannot inject text: %s", self.last_error)
            return False

        # Save current clipboard
        old_clip = self._get_clipboard()

        # Copy text to clipboard
        if not self._run(["wl-copy", "--", text]):
            self.last_error = "wl-copy failed"
            return False

        time.sleep(0.05)

        # Paste via ydotool
        ok = self._run(["ydotool", "key"] + self._paste_keys)
        if not ok:
            self.last_error = "ydotool key simulation failed (is ydotoold running?)"

        # Restore clipboard after a short delay
        if old_clip is not None:
            time.sleep(0.15)
            self._run(["wl-copy", "--", old_clip])

        if ok:
            log.info("Text injected (%d chars)", len(text))
        return ok

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
