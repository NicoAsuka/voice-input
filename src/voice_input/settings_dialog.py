# src/voice_input/settings_dialog.py
from __future__ import annotations

import asyncio
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from voice_input.asr.model_manager import REGISTRY
from voice_input.config import AppConfig, save_config
from voice_input.keyring_helper import get_secret, set_secret

log = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Voice Input settings dialog following Breeze style conventions."""

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Voice Input — Settings")
        self.setMinimumWidth(420)
        self._config = config
        self._build_ui()
        self._load_from_config()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # LLM tab
        llm_tab = QWidget()
        llm_layout = QVBoxLayout(llm_tab)
        form = QFormLayout()

        self._api_base_edit = QLineEdit()
        self._api_base_edit.setPlaceholderText("https://api.openai.com/v1")
        form.addRow("API Base URL:", self._api_base_edit)

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

        llm_layout.addLayout(form)

        test_layout = QHBoxLayout()
        test_layout.addStretch()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._on_test)
        test_layout.addWidget(self._test_btn)
        llm_layout.addLayout(test_layout)
        tabs.addTab(llm_tab, "LLM")

        # STT tab
        stt_tab = QWidget()
        stt_layout = QVBoxLayout(stt_tab)

        stt_layout.addWidget(QLabel("OpenAI Whisper API"))
        openai_form = QFormLayout()
        self._stt_openai_base_edit = QLineEdit()
        self._stt_openai_base_edit.setPlaceholderText("https://api.openai.com/v1")
        openai_form.addRow("API Base:", self._stt_openai_base_edit)
        self._stt_openai_model_edit = QLineEdit()
        self._stt_openai_model_edit.setPlaceholderText("whisper-1")
        openai_form.addRow("Model:", self._stt_openai_model_edit)
        self._stt_openai_key_edit = QLineEdit()
        self._stt_openai_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._stt_openai_key_edit.setPlaceholderText("sk-...")
        openai_form.addRow("API Key:", self._stt_openai_key_edit)
        stt_layout.addLayout(openai_form)

        stt_layout.addWidget(QLabel("Google Speech-to-Text"))
        google_form = QFormLayout()
        self._stt_google_creds_edit = QLineEdit()
        self._stt_google_creds_edit.setPlaceholderText("/path/to/credentials.json")
        google_form.addRow("Credentials:", self._stt_google_creds_edit)
        stt_layout.addLayout(google_form)

        stt_layout.addWidget(QLabel("字节火山语音识别"))
        volc_form = QFormLayout()
        self._stt_volc_appid_edit = QLineEdit()
        self._stt_volc_appid_edit.setPlaceholderText("app-id")
        volc_form.addRow("App ID:", self._stt_volc_appid_edit)
        self._stt_volc_resource_id_edit = QLineEdit()
        self._stt_volc_resource_id_edit.setPlaceholderText("volc.bigasr.sauc.concurrent")
        volc_form.addRow("Resource ID:", self._stt_volc_resource_id_edit)
        self._stt_volc_ak_edit = QLineEdit()
        self._stt_volc_ak_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._stt_volc_ak_edit.setPlaceholderText("access key")
        volc_form.addRow("Access Key:", self._stt_volc_ak_edit)
        self._stt_volc_sk_edit = QLineEdit()
        self._stt_volc_sk_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._stt_volc_sk_edit.setPlaceholderText("secret key")
        volc_form.addRow("Secret Key:", self._stt_volc_sk_edit)
        stt_layout.addLayout(volc_form)

        stt_layout.addStretch()
        tabs.addTab(stt_tab, "STT")

        # Local (sherpa-onnx) tab
        local_tab = QWidget()
        local_layout = QVBoxLayout(local_tab)
        local_form = QFormLayout()

        self._sherpa_model_combo = QComboBox()
        for model_id in REGISTRY:
            meta = REGISTRY[model_id]
            label = f"{model_id}  ({meta.language}, {meta.size_bytes // 1_000_000}MB)"
            self._sherpa_model_combo.addItem(label, model_id)
        local_form.addRow("Model:", self._sherpa_model_combo)

        self._sherpa_vad_check = QCheckBox("启用 VAD（去除静音）")
        local_form.addRow("VAD:", self._sherpa_vad_check)

        self._sherpa_threads_spin = QSpinBox()
        self._sherpa_threads_spin.setRange(1, 8)
        local_form.addRow("Threads:", self._sherpa_threads_spin)

        self._sherpa_provider_combo = QComboBox()
        for provider in ("cpu", "cuda", "coreml"):
            self._sherpa_provider_combo.addItem(provider, provider)
        local_form.addRow("Provider:", self._sherpa_provider_combo)

        local_layout.addLayout(local_form)
        local_layout.addStretch()
        tabs.addTab(local_tab, "Local")

        layout.addWidget(tabs)

        # Button box
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_from_config(self) -> None:
        # LLM
        llm = self._config.get("llm", {})
        self._api_base_edit.setText(llm.get("api_base", "https://api.openai.com/v1"))
        self._model_edit.setText(llm.get("model", "gpt-4o-mini"))
        self._load_api_key()

        # STT
        stt = self._config.get("stt", {})
        openai_cfg = stt.get("openai", {})
        self._stt_openai_base_edit.setText(
            openai_cfg.get("api_base", "https://api.openai.com/v1")
        )
        self._stt_openai_model_edit.setText(openai_cfg.get("model", "whisper-1"))
        self._load_keyring_field(self._stt_openai_key_edit, "stt-openai-api-key")

        google_cfg = stt.get("google", {})
        self._stt_google_creds_edit.setText(google_cfg.get("credentials_path", ""))

        volc_cfg = stt.get("volcengine", {})
        self._stt_volc_appid_edit.setText(volc_cfg.get("app_id", ""))
        self._stt_volc_resource_id_edit.setText(
            volc_cfg.get("resource_id", "volc.seedasr.sauc.duration")
        )
        self._load_keyring_field(self._stt_volc_ak_edit, "stt-volcengine-access-key")
        self._load_keyring_field(self._stt_volc_sk_edit, "stt-volcengine-secret-key")

        # Local (sherpa-onnx)
        sherpa_cfg = stt.get("sherpa", {})
        current_model = sherpa_cfg.get("model_id", "sherpa-onnx-paraformer-zh-2024-03-09")
        idx = self._sherpa_model_combo.findData(current_model)
        if idx >= 0:
            self._sherpa_model_combo.setCurrentIndex(idx)
        self._sherpa_vad_check.setChecked(sherpa_cfg.get("vad_enabled", True))
        self._sherpa_threads_spin.setValue(sherpa_cfg.get("num_threads", 2))
        current_provider = sherpa_cfg.get("provider", "cpu")
        idx = self._sherpa_provider_combo.findData(current_provider)
        if idx >= 0:
            self._sherpa_provider_combo.setCurrentIndex(idx)

    def _load_api_key(self) -> None:
        key = get_secret("llm-api-key")
        if key:
            self._api_key_edit.setText(key)

    def _save_api_key(self, key: str) -> None:
        set_secret("llm-api-key", key)

    def _load_keyring_field(self, field: QLineEdit, key: str) -> None:
        value = get_secret(key)
        if value:
            field.setText(value)

    def _save_keyring_field(self, field: QLineEdit, key: str) -> None:
        set_secret(key, field.text().strip())

    def _toggle_key_visibility(self, checked: bool) -> None:
        if checked:
            self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self._toggle_vis_btn.setText("Hide")
        else:
            self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self._toggle_vis_btn.setText("Show")

    def _on_save(self) -> None:
        self._config["llm"]["api_base"] = (
            self._api_base_edit.text().strip() or "https://api.openai.com/v1"
        )
        self._config["llm"]["model"] = self._model_edit.text().strip() or "gpt-4o-mini"
        self._save_api_key(self._api_key_edit.text().strip())

        stt = self._config.setdefault("stt", {})
        openai_cfg = stt.setdefault("openai", {})
        openai_cfg["api_base"] = (
            self._stt_openai_base_edit.text().strip() or "https://api.openai.com/v1"
        )
        openai_cfg["model"] = (
            self._stt_openai_model_edit.text().strip() or "whisper-1"
        )
        self._save_keyring_field(self._stt_openai_key_edit, "stt-openai-api-key")

        google_cfg = stt.setdefault("google", {})
        google_cfg["credentials_path"] = self._stt_google_creds_edit.text().strip()

        volc_cfg = stt.setdefault("volcengine", {})
        volc_cfg["app_id"] = self._stt_volc_appid_edit.text().strip()
        volc_cfg["resource_id"] = (
            self._stt_volc_resource_id_edit.text().strip()
            or "volc.seedasr.sauc.duration"
        )
        self._save_keyring_field(self._stt_volc_ak_edit, "stt-volcengine-access-key")
        self._save_keyring_field(self._stt_volc_sk_edit, "stt-volcengine-secret-key")

        sherpa_cfg = stt.setdefault("sherpa", {})
        sherpa_cfg["model_id"] = self._sherpa_model_combo.currentData()
        sherpa_cfg["vad_enabled"] = self._sherpa_vad_check.isChecked()
        sherpa_cfg["num_threads"] = self._sherpa_threads_spin.value()
        sherpa_cfg["provider"] = self._sherpa_provider_combo.currentData()

        save_config(self._config)
        self.accept()

    def _on_test(self) -> None:
        """Test API connection."""
        from voice_input.postprocess.llm import LLMRefiner

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
