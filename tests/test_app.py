# tests/test_app.py
from unittest.mock import MagicMock, patch
from voice_input.app import AppState


def test_state_transitions():
    """Verify valid state transitions."""
    assert AppState.IDLE.can_transition_to(AppState.RECORDING)
    assert AppState.RECORDING.can_transition_to(AppState.REFINING)
    assert AppState.RECORDING.can_transition_to(AppState.IDLE)
    assert AppState.REFINING.can_transition_to(AppState.IDLE)
    assert not AppState.IDLE.can_transition_to(AppState.REFINING)
    assert not AppState.REFINING.can_transition_to(AppState.RECORDING)
