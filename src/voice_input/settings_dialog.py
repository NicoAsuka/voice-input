# src/voice_input/settings_dialog.py
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from voice_input.config import AppConfig, save_config, xdg_config_dir

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

log = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """LLM settings dialog following Breeze style conventions."""

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Voice Input — LLM Settings")
        self.setMinimumWidth(420)
        self._config = config
        self._build_ui()
        self._load_from_config()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        # API Base URL
        self._api_base_edit = QLineEdit()
        self._api_base_edit.setPlaceholderText("https://api.openai.com/v1")
        form.addRow("API Base URL:", self._api_base_edit)

        # API Key with visibility toggle
        key_layout = QHBoxLayout()
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("sk-...")
        key_layout.addWidget(self._api_key_edit)
        self._toggle_vis_btn = QPushButton("Show")
        self._toggle_vis_btn.setFixedWidth(60)
        self._toggle_vis_btn.setCheckable(True)
        self._toggle_vis_btn.toggled.connect(self._toggle_key_visibility)
        key_layout.addWidget(self._toggle_vis_btn)
        form.addRow("API Key:", key_layout)

        # Model
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("gpt-4o-mini")
        form.addRow("Model:", self._model_edit)

        layout.addLayout(form)

        # Test button
        test_layout = QHBoxLayout()
        test_layout.addStretch()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._on_test)
        test_layout.addWidget(self._test_btn)
        layout.addLayout(test_layout)

        # Button box
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_from_config(self) -> None:
        llm = self._config.get("llm", {})
        self._api_base_edit.setText(llm.get("api_base", "https://api.openai.com/v1"))
        self._model_edit.setText(llm.get("model", "gpt-4o-mini"))
        # API key loaded from keyring, not config
        self._load_api_key()

    def _load_api_key(self) -> None:
        try:
            import keyring
            key = keyring.get_password("voice-input", "llm-api-key")
            if key:
                self._api_key_edit.setText(key)
        except Exception as e:
            log.debug("Could not load API key from keyring: %s", e)

    def _save_api_key(self, key: str) -> None:
        try:
            import keyring
            if key:
                keyring.set_password("voice-input", "llm-api-key", key)
            else:
                keyring.delete_password("voice-input", "llm-api-key")
        except Exception as e:
            log.warning("Could not save API key to keyring: %s", e)

    def _toggle_key_visibility(self, checked: bool) -> None:
        if checked:
            self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self._toggle_vis_btn.setText("Hide")
        else:
            self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self._toggle_vis_btn.setText("Show")

    def _on_save(self) -> None:
        self._config["llm"]["api_base"] = self._api_base_edit.text().strip() or "https://api.openai.com/v1"
        self._config["llm"]["model"] = self._model_edit.text().strip() or "gpt-4o-mini"
        save_config(self._config)
        self._save_api_key(self._api_key_edit.text().strip())
        self.accept()

    def _on_test(self) -> None:
        """Test API connection."""
        from voice_input.llm import LLMRefiner

        api_base = self._api_base_edit.text().strip() or "https://api.openai.com/v1"
        api_key = self._api_key_edit.text().strip()
        model = self._model_edit.text().strip() or "gpt-4o-mini"

        if not api_key:
            QMessageBox.warning(self, "Test", "API Key is required.")
            return

        self._test_btn.setEnabled(False)
        self._test_btn.setText("Testing...")

        async def do_test():
            refiner = LLMRefiner(api_base=api_base, api_key=api_key, model=model)
            try:
                success, msg = await refiner.test_connection()
                if success:
                    QMessageBox.information(self, "Test", f"Success: {msg}")
                else:
                    QMessageBox.warning(self, "Test", f"Failed: {msg}")
            finally:
                await refiner.close()
                self._test_btn.setEnabled(True)
                self._test_btn.setText("Test Connection")

        try:
            loop = asyncio.get_event_loop()
            loop.create_task(do_test())
        except RuntimeError:
            QMessageBox.warning(self, "Test", "Async event loop not available.")
            self._test_btn.setEnabled(True)
            self._test_btn.setText("Test Connection")

    def get_api_key(self) -> str:
        return self._api_key_edit.text().strip()
