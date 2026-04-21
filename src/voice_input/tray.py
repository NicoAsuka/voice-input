# src/voice_input/tray.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtGui import QAction, QActionGroup, QIcon
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

log = logging.getLogger(__name__)

LANGUAGES = {
    "en": "English",
    "zh": "简体中文",
    "zh-TW": "繁體中文",
    "ja": "日本語",
    "ko": "한국어",
}

STT_BACKENDS = {
    "local": "Local",
    "openai": "OpenAI Whisper API",
    "google": "Google Speech-to-Text",
    "volcengine": "字节火山语音识别",
}

ENGINES = {
    "whisper": "faster-whisper",
    "sensevoice": "SenseVoice Small",
}


class TrayManager(QSystemTrayIcon):
    """System tray icon using StatusNotifierItem (via Qt6 on KDE).

    Provides context menu for controlling the voice input app.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = "Idle"
        self._current_language = "zh"
        self._current_backend = "local"
        self._llm_enabled = True

        # Icons from Breeze theme
        self._icon_idle = QIcon.fromTheme("audio-input-microphone")
        self._icon_recording = QIcon.fromTheme("audio-input-microphone")
        self._icon_muted = QIcon.fromTheme("audio-input-microphone-muted")

        self.setIcon(self._icon_idle)
        self.setToolTip("Voice Input — Idle")

        self._build_menu()

    def _build_menu(self) -> None:
        menu = QMenu()

        # Status
        self._status_action = QAction("Status: Idle", menu)
        self._status_action.setEnabled(False)
        menu.addAction(self._status_action)
        menu.addSeparator()

        # Start/Stop
        self._toggle_action = QAction("Start Recording", menu)
        menu.addAction(self._toggle_action)
        menu.addSeparator()

        # Language submenu
        lang_menu = QMenu("Language", menu)
        self._lang_group = QActionGroup(lang_menu)
        self._lang_group.setExclusive(True)
        self._lang_actions: dict[str, QAction] = {}
        for code, name in LANGUAGES.items():
            action = QAction(name, lang_menu)
            action.setCheckable(True)
            action.setData(code)
            if code == self._current_language:
                action.setChecked(True)
            self._lang_group.addAction(action)
            lang_menu.addAction(action)
            self._lang_actions[code] = action
        menu.addMenu(lang_menu)

        # STT Backend submenu
        stt_menu = QMenu("STT Backend", menu)
        self._stt_group = QActionGroup(stt_menu)
        self._stt_group.setExclusive(True)
        self._stt_actions: dict[str, QAction] = {}
        for key, label in STT_BACKENDS.items():
            action = QAction(label, stt_menu)
            action.setCheckable(True)
            action.setData(key)
            if key == self._current_backend:
                action.setChecked(True)
            self._stt_group.addAction(action)
            stt_menu.addAction(action)
            self._stt_actions[key] = action
        stt_menu.addSeparator()
        self._stt_settings_action = QAction("Settings...", stt_menu)
        stt_menu.addAction(self._stt_settings_action)
        menu.addMenu(stt_menu)

        # Local Engine submenu. This setting only affects the local STT backend.
        engine_menu = QMenu("Local Engine", menu)
        self._engine_group = QActionGroup(engine_menu)
        self._engine_group.setExclusive(True)
        self._engine_actions: dict[str, QAction] = {}
        for key, label in ENGINES.items():
            action = QAction(label, engine_menu)
            action.setCheckable(True)
            action.setData(key)
            if key == "whisper":
                action.setChecked(True)
            self._engine_group.addAction(action)
            engine_menu.addAction(action)
            self._engine_actions[key] = action
        menu.addMenu(engine_menu)

        # LLM submenu
        llm_menu = QMenu("LLM Refinement", menu)
        self._llm_toggle = QAction("Enabled", llm_menu)
        self._llm_toggle.setCheckable(True)
        self._llm_toggle.setChecked(self._llm_enabled)
        llm_menu.addAction(self._llm_toggle)
        self._llm_settings_action = QAction("Settings...", llm_menu)
        llm_menu.addAction(self._llm_settings_action)
        menu.addMenu(llm_menu)

        menu.addSeparator()

        # Preferences
        self._prefs_action = QAction("Preferences...", menu)
        menu.addAction(self._prefs_action)

        # About
        self._about_action = QAction("About", menu)
        menu.addAction(self._about_action)

        menu.addSeparator()

        # Quit
        self._quit_action = QAction("Quit", menu)
        self._quit_action.triggered.connect(QApplication.quit)
        menu.addAction(self._quit_action)

        self.setContextMenu(menu)

    # --- Public interface for AppController to connect signals ---

    @property
    def toggle_action(self) -> QAction:
        return self._toggle_action

    @property
    def lang_group(self) -> QActionGroup:
        return self._lang_group

    @property
    def stt_group(self) -> QActionGroup:
        return self._stt_group

    @property
    def stt_settings_action(self) -> QAction:
        return self._stt_settings_action

    @property
    def engine_group(self) -> QActionGroup:
        return self._engine_group

    @property
    def llm_toggle(self) -> QAction:
        return self._llm_toggle

    @property
    def llm_settings_action(self) -> QAction:
        return self._llm_settings_action

    @property
    def prefs_action(self) -> QAction:
        return self._prefs_action

    @property
    def about_action(self) -> QAction:
        return self._about_action

    def set_state(self, state: str) -> None:
        """Update tray state: 'Idle', 'Recording', 'Transcribing', 'Refining'."""
        self._state = state
        self._status_action.setText(f"Status: {state}")
        self.setToolTip(f"Voice Input — {state}")

        if state == "Recording":
            self._toggle_action.setText("Stop Recording")
            self._toggle_action.setEnabled(True)
            self.setIcon(self._icon_recording)
        elif state in ("Transcribing", "Refining"):
            self._toggle_action.setText("Stop Recording")
            self._toggle_action.setEnabled(False)
        else:
            self._toggle_action.setText("Start Recording")
            self._toggle_action.setEnabled(True)
            self.setIcon(self._icon_idle)

    def set_language(self, code: str) -> None:
        self._current_language = code
        if code in self._lang_actions:
            self._lang_actions[code].setChecked(True)

    def set_backend(self, backend: str) -> None:
        self._current_backend = backend
        if backend in self._stt_actions:
            self._stt_actions[backend].setChecked(True)

    def set_engine(self, engine: str) -> None:
        if engine in self._engine_actions:
            self._engine_actions[engine].setChecked(True)

    def set_llm_enabled(self, enabled: bool) -> None:
        self._llm_enabled = enabled
        self._llm_toggle.setChecked(enabled)
