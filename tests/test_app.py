# tests/test_app.py
from unittest.mock import AsyncMock, MagicMock

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
