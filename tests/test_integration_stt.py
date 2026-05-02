"""Integration tests for STT backend switching."""
import copy
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from voice_input.backends import create_backend
from voice_input.backends.base import TranscriptionBackend
from voice_input.config import DEFAULT_CONFIG


def test_default_config_creates_sherpa_backend():
    """Default config should create a sherpa backend."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    backend = create_backend(cfg)
    assert isinstance(backend, TranscriptionBackend)
    desc = backend.describe()
    assert "sherpa" in desc.model_id or "paraformer" in desc.model_id


def test_all_backends_implement_interface():
    """Every backend returned by the factory implements TranscriptionBackend."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    cfg["stt"]["backend"] = "sherpa"
    sherpa = create_backend(cfg)
    assert isinstance(sherpa, TranscriptionBackend)
    assert hasattr(sherpa, "describe")
    assert hasattr(sherpa, "create_session")

    cfg["stt"]["backend"] = "openai"
    with patch("keyring.get_password", return_value="sk-test"):
        openai_b = create_backend(cfg)
    assert isinstance(openai_b, TranscriptionBackend)
    assert hasattr(openai_b, "create_session")

    cfg["stt"]["backend"] = "google"
    google_b = create_backend(cfg)
    assert isinstance(google_b, TranscriptionBackend)
    assert hasattr(google_b, "create_session")

    cfg["stt"]["backend"] = "volcengine"
    with patch("keyring.get_password", return_value="fake"):
        volc_b = create_backend(cfg)
    assert isinstance(volc_b, TranscriptionBackend)
    assert hasattr(volc_b, "create_session")
