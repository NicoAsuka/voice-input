from PyQt6.QtWidgets import QApplication

from voice_input.settings_dialog import SettingsDialog

_QAPP = None


def _app():
    global _QAPP
    _QAPP = QApplication.instance() or QApplication([])
    return _QAPP


def test_settings_dialog_loads_and_saves_volcengine_resource_id(monkeypatch):
    _app()
    saved = []
    monkeypatch.setattr("voice_input.settings_dialog.save_config", saved.append)
    monkeypatch.setattr(SettingsDialog, "_load_api_key", lambda self: None)
    monkeypatch.setattr(SettingsDialog, "_load_keyring_field", lambda self, field, key: None)
    monkeypatch.setattr(SettingsDialog, "_save_keyring_field", lambda self, field, key: None)

    config = {
        "llm": {},
        "stt": {
            "openai": {},
            "google": {},
            "volcengine": {
                "app_id": "app-id",
                "resource_id": "volc.bigasr.sauc.concurrent",
            },
        },
    }
    dialog = SettingsDialog(config)

    assert dialog._stt_volc_resource_id_edit.text() == "volc.bigasr.sauc.concurrent"

    dialog._stt_volc_resource_id_edit.setText("volc.seedasr.sauc.duration")
    dialog._on_save()

    assert saved[0]["stt"]["volcengine"]["resource_id"] == "volc.seedasr.sauc.duration"


def test_settings_dialog_saves_sherpa_config(monkeypatch):
    _app()
    saved = []
    monkeypatch.setattr("voice_input.settings_dialog.save_config", saved.append)
    monkeypatch.setattr(SettingsDialog, "_load_api_key", lambda self: None)
    monkeypatch.setattr(SettingsDialog, "_load_keyring_field", lambda self, field, key: None)
    monkeypatch.setattr(SettingsDialog, "_save_keyring_field", lambda self, field, key: None)

    config = {
        "llm": {},
        "stt": {
            "openai": {},
            "google": {},
            "volcengine": {},
        },
    }
    dialog = SettingsDialog(config)

    # Check defaults
    assert dialog._sherpa_model_combo.currentData() == "sherpa-onnx-paraformer-zh-2024-03-09"
    assert dialog._sherpa_vad_check.isChecked() is True
    assert dialog._sherpa_threads_spin.value() == 2
    assert dialog._sherpa_provider_combo.currentData() == "cpu"

    # Modify values
    dialog._sherpa_model_combo.setCurrentIndex(1)
    dialog._sherpa_vad_check.setChecked(False)
    dialog._sherpa_threads_spin.setValue(4)
    dialog._sherpa_provider_combo.setCurrentIndex(1)

    dialog._on_save()

    sherpa = saved[0]["stt"]["sherpa"]
    assert sherpa["model_id"] == "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    assert sherpa["vad_enabled"] is False
    assert sherpa["num_threads"] == 4
    assert sherpa["provider"] == "cuda"


def test_settings_dialog_loads_sherpa_config(monkeypatch):
    _app()
    monkeypatch.setattr(SettingsDialog, "_load_api_key", lambda self: None)
    monkeypatch.setattr(SettingsDialog, "_load_keyring_field", lambda self, field, key: None)

    config = {
        "llm": {},
        "stt": {
            "openai": {},
            "google": {},
            "volcengine": {},
            "sherpa": {
                "model_id": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                "vad_enabled": False,
                "num_threads": 6,
                "provider": "cuda",
            },
        },
    }
    dialog = SettingsDialog(config)

    assert dialog._sherpa_model_combo.currentData() == "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    assert dialog._sherpa_vad_check.isChecked() is False
    assert dialog._sherpa_threads_spin.value() == 6
    assert dialog._sherpa_provider_combo.currentData() == "cuda"
