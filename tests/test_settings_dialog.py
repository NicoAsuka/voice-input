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
