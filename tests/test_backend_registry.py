from __future__ import annotations

from voice_input.backends.registry import RegistryState, compute_signature


def test_registry_state_enum_values():
    assert RegistryState.LOADING.value == "loading"
    assert RegistryState.READY.value == "ready"
    assert RegistryState.RELOADING.value == "reloading"
    assert RegistryState.ERROR.value == "error"


def test_compute_signature_same_for_same_config():
    cfg1 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "x"}}}
    cfg2 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "x"}}}
    assert compute_signature(cfg1) == compute_signature(cfg2)


def test_compute_signature_changes_when_model_changes():
    cfg1 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "x"}}}
    cfg2 = {"stt": {"backend": "sherpa", "sherpa": {"model_id": "y"}}}
    assert compute_signature(cfg1) != compute_signature(cfg2)


def test_compute_signature_changes_when_backend_changes():
    cfg1 = {"stt": {"backend": "sherpa"}}
    cfg2 = {"stt": {"backend": "volcengine"}}
    assert compute_signature(cfg1) != compute_signature(cfg2)


def test_compute_signature_ignores_unrelated_fields():
    cfg1 = {"stt": {"backend": "sherpa"}, "ui": {"x": 1}}
    cfg2 = {"stt": {"backend": "sherpa"}, "ui": {"x": 2}}
    assert compute_signature(cfg1) == compute_signature(cfg2)
