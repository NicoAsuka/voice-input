from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from voice_input.app import AppController, AppState, _can_transition
from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    Session,
    TranscriptionBackend,
)
from voice_input.backends.registry import BackendRegistry, RegistryState


def test_state_transitions():
    assert _can_transition(AppState.IDLE, AppState.RECORDING)
    assert _can_transition(AppState.RECORDING, AppState.TRANSCRIBING)
    assert _can_transition(AppState.RECORDING, AppState.IDLE)
    assert _can_transition(AppState.TRANSCRIBING, AppState.IDLE)
    assert _can_transition(AppState.TRANSCRIBING, AppState.REFINING)
    assert _can_transition(AppState.REFINING, AppState.IDLE)
    assert not _can_transition(AppState.IDLE, AppState.REFINING)
    assert not _can_transition(AppState.REFINING, AppState.RECORDING)


def test_transcribing_state_exists():
    assert hasattr(AppState, "TRANSCRIBING")
    assert AppState.TRANSCRIBING.value == "Transcribing"


def test_llm_toggle_off_closes_existing_refiner():
    controller = AppController.__new__(AppController)
    controller._llm_enabled = True
    refiner = MagicMock()
    controller._llm = refiner
    controller._pipeline = MagicMock()
    controller._config = {"postprocess": {"enabled": True}}

    with patch("voice_input.app.save_config"), patch(
        "voice_input.app.asyncio.ensure_future"
    ) as ensure_future:
        AppController._on_llm_toggled(controller, False)

    assert controller._llm_enabled is False
    assert controller._llm is None
    assert controller._pipeline is None
    refiner.close.assert_called_once()
    ensure_future.assert_called_once()


def test_backend_changed_triggers_synchronize():
    controller = AppController.__new__(AppController)
    controller._config = {"stt": {"backend": "sherpa"}}
    controller._registry = MagicMock()
    action = MagicMock()
    action.data.return_value = "openai"

    with patch("voice_input.app.save_config"), patch(
        "voice_input.app.asyncio.ensure_future"
    ) as ef:
        AppController._on_backend_changed(controller, action)

    assert controller._config["stt"]["backend"] == "openai"
    ef.assert_called_once()


def test_scene_changed_updates_config():
    controller = AppController.__new__(AppController)
    controller._config = {"postprocess": {}}
    controller._scenes = MagicMock()
    action = MagicMock()
    action.data.return_value = "code"

    with patch("voice_input.app.save_config"):
        AppController._on_scene_changed(controller, action)

    controller._scenes.set_active.assert_called_once_with("code")
    assert controller._config["postprocess"]["active_scene"] == "code"


def test_start_recording_blocked_when_registry_not_ready():
    controller = AppController.__new__(AppController)
    controller._state = AppState.IDLE
    controller._registry = MagicMock()
    controller._registry.is_ready.return_value = False
    controller._registry.state.return_value = RegistryState.LOADING
    controller._send_notification = MagicMock()

    AppController._on_start_recording(controller)

    controller._send_notification.assert_called_once()
    assert controller._state == AppState.IDLE
