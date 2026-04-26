# tests/test_app.py
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from voice_input.app import AppController, AppState


def test_state_transitions():
    """Verify valid state transitions."""
    assert AppState.IDLE.can_transition_to(AppState.RECORDING)
    assert AppState.RECORDING.can_transition_to(AppState.REFINING)
    assert AppState.RECORDING.can_transition_to(AppState.IDLE)
    assert AppState.REFINING.can_transition_to(AppState.IDLE)
    assert not AppState.IDLE.can_transition_to(AppState.REFINING)
    assert not AppState.REFINING.can_transition_to(AppState.RECORDING)


def test_transcribing_state_exists():
    assert hasattr(AppState, "TRANSCRIBING")
    assert AppState.TRANSCRIBING.value == "Transcribing"


def test_transcribing_state_transitions():
    """Verify TRANSCRIBING state transitions."""
    assert AppState.RECORDING.can_transition_to(AppState.TRANSCRIBING)
    assert AppState.TRANSCRIBING.can_transition_to(AppState.IDLE)
    assert AppState.TRANSCRIBING.can_transition_to(AppState.REFINING)
    assert not AppState.IDLE.can_transition_to(AppState.TRANSCRIBING)
    assert not AppState.TRANSCRIBING.can_transition_to(AppState.RECORDING)


def test_backend_worker_restart_rejected_before_worker_ready():
    controller = AppController.__new__(AppController)
    controller._worker_ready = False
    controller._restart_backend_worker = MagicMock()
    controller._send_notification = MagicMock()
    controller._tray = MagicMock()
    controller._state = AppState.IDLE
    controller._config = {"stt": {"backend": "local"}}
    action = MagicMock()
    action.data.return_value = "openai"

    AppController._on_backend_changed(controller, action)

    assert controller._config["stt"]["backend"] == "local"
    controller._restart_backend_worker.assert_not_called()
    controller._send_notification.assert_called_once()


def test_engine_change_sets_sensevoice_default_model():
    controller = AppController.__new__(AppController)
    controller._worker_ready = True
    controller._restart_backend_worker = MagicMock()
    controller._state = AppState.IDLE
    controller._config = {
        "stt": {
            "backend": "local",
            "local": {"engine": "whisper", "model": "medium"},
        },
    }
    action = MagicMock()
    action.data.return_value = "sensevoice"

    with (
        patch("voice_input.app.save_config"),
        patch("voice_input.app.importlib.util.find_spec", return_value=object()),
    ):
        AppController._on_engine_changed(controller, action)

    assert controller._config["stt"]["local"]["engine"] == "sensevoice"
    assert controller._config["stt"]["local"]["model"] == "iic/SenseVoiceSmall"
    controller._restart_backend_worker.assert_called_once()


def test_engine_change_rejected_when_sensevoice_dependency_missing():
    controller = AppController.__new__(AppController)
    controller._worker_ready = True
    controller._restart_backend_worker = MagicMock()
    controller._send_notification = MagicMock()
    controller._tray = MagicMock()
    controller._state = AppState.IDLE
    controller._config = {
        "stt": {
            "backend": "local",
            "local": {"engine": "whisper", "model": "small"},
        },
    }
    action = MagicMock()
    action.data.return_value = "sensevoice"

    with patch("voice_input.app.importlib.util.find_spec", return_value=None):
        AppController._on_engine_changed(controller, action)

    assert controller._config["stt"]["local"]["engine"] == "whisper"
    assert controller._config["stt"]["local"]["model"] == "small"
    controller._restart_backend_worker.assert_not_called()
    controller._send_notification.assert_called_once()


@pytest.mark.asyncio
async def test_final_transcription_uses_full_buffer_before_injecting():
    controller = AppController.__new__(AppController)
    controller._backend = MagicMock()
    controller._backend.transcribe = AsyncMock(return_value="final text")
    controller._language = "zh"
    controller._llm_enabled = False
    controller._llm = None
    controller._last_transcription = "partial text"
    controller._inject_and_finish = MagicMock()

    audio = np.ones(32000, dtype=np.int16)
    await AppController._transcribe_and_inject(
        controller,
        audio,
        fallback_text="partial text",
    )

    controller._backend.transcribe.assert_awaited_once_with(audio, "zh")
    assert controller._last_transcription == "final text"
    controller._inject_and_finish.assert_called_once_with("final text")


@pytest.mark.asyncio
async def test_final_transcription_falls_back_to_partial_text():
    controller = AppController.__new__(AppController)
    controller._backend = MagicMock()
    controller._backend.transcribe = AsyncMock(return_value="")
    controller._language = "zh"
    controller._llm_enabled = False
    controller._llm = None
    controller._last_transcription = "partial text"
    controller._inject_and_finish = MagicMock()

    audio = np.ones(32000, dtype=np.int16)
    await AppController._transcribe_and_inject(
        controller,
        audio,
        fallback_text="partial text",
    )

    controller._inject_and_finish.assert_called_once_with("partial text")
