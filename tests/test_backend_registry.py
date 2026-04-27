from __future__ import annotations

import asyncio
import time

import pytest

from voice_input.backends.base import (
    BackendCapabilities,
    BackendDescriptor,
    Session,
    TranscriptionBackend,
)
from voice_input.backends.registry import (
    BackendRegistry,
    RegistryState,
    compute_signature,
)


class FakeSession(Session):
    def __init__(self, backend_ref: "FakeBackend") -> None:
        self._backend = backend_ref
        self.pushed: list = []
        self.finished = False

    def push_audio(self, pcm_int16):
        self.pushed.append(pcm_int16)

    async def finish(self) -> str:
        self.finished = True
        return "fake-text"

    def cancel(self) -> None:
        pass


class FakeBackend(TranscriptionBackend):
    """可控的测试 backend。"""

    def __init__(
        self,
        *,
        init_delay: float = 0.0,
        fail_init: bool = False,
        fail_describe: bool = False,
        model_id: str = "m",
    ) -> None:
        self._init_delay = init_delay
        self._fail_init = fail_init
        self._fail_describe = fail_describe
        self._model_id = model_id
        self._ready = False
        self.shutdown_called = False
        self.init_call_count = 0

    async def initialize(self) -> None:
        self.init_call_count += 1
        if self._init_delay:
            await asyncio.sleep(self._init_delay)
        if self._fail_init:
            raise RuntimeError("init boom")
        self._ready = True

    def describe(self) -> BackendDescriptor:
        if self._fail_describe:
            raise RuntimeError("describe boom")
        return BackendDescriptor(
            backend_id="fake",
            model_id=self._model_id,
            capabilities=BackendCapabilities(),
        )

    def create_session(self, language: str) -> Session:
        return FakeSession(self)

    async def shutdown(self) -> None:
        self.shutdown_called = True

    def is_ready(self) -> bool:
        return self._ready


def make_factory(backend: TranscriptionBackend):
    def factory(config):
        return backend

    return factory


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


def test_compute_signature_treats_missing_or_null_stt_as_empty():
    cfg1 = {}
    cfg2 = {"stt": None}
    assert compute_signature(cfg1) == compute_signature(cfg2)


@pytest.mark.asyncio
async def test_start_returns_immediately_while_loading():
    slow_backend = FakeBackend(init_delay=0.05)
    reg = BackendRegistry({"stt": {"backend": "fake"}}, factory=make_factory(slow_backend))

    t0 = time.monotonic()
    await reg.start()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.02
    assert reg.state() == RegistryState.LOADING
    assert reg.is_ready() is False
    assert reg.current_descriptor() is None

    await asyncio.sleep(0.08)
    assert reg.state() == RegistryState.READY
    assert reg.is_ready() is True
    assert reg.current_descriptor() is not None
    assert reg.current_descriptor().backend_id == "fake"

    await reg.shutdown()


@pytest.mark.asyncio
async def test_create_session_raises_when_not_ready():
    slow_backend = FakeBackend(init_delay=0.2)
    reg = BackendRegistry({"stt": {"backend": "fake"}}, factory=make_factory(slow_backend))

    await reg.start()
    with pytest.raises(RuntimeError, match="not ready"):
        reg.create_session("zh")

    await reg.shutdown()


@pytest.mark.asyncio
async def test_create_session_works_when_ready():
    backend = FakeBackend()
    reg = BackendRegistry({"stt": {"backend": "fake"}}, factory=make_factory(backend))

    await reg.start()
    await asyncio.sleep(0.01)

    session = reg.create_session("zh")
    assert isinstance(session, FakeSession)

    await reg.shutdown()


@pytest.mark.asyncio
async def test_reload_failure_sets_error_without_effective_backend():
    bad = FakeBackend(fail_init=True)
    reg = BackendRegistry({"stt": {"backend": "fake"}}, factory=make_factory(bad))

    await reg.start()
    await asyncio.sleep(0.01)

    assert reg.is_ready() is False
    assert reg.state() == RegistryState.ERROR
    assert "init boom" in (reg.last_error() or "")
    assert reg.current_descriptor() is None

    await reg.shutdown()


@pytest.mark.asyncio
async def test_reload_worker_keeps_old_effective_until_new_is_ready():
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2", init_delay=0.05)
    next_backend = [backend1, backend2]

    def factory(config):
        return next_backend.pop(0)

    cfg1 = {"stt": {"backend": "fake", "fake": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "fake", "fake": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=factory)

    await reg.start()
    await asyncio.sleep(0.01)
    assert reg.current_descriptor().model_id == "m1"

    reload_task = asyncio.create_task(
        reg._reload_worker(cfg2, compute_signature(cfg2))
    )
    await asyncio.sleep(0.01)
    assert reg.is_ready() is True
    assert reg.state() == RegistryState.READY
    assert reg.current_descriptor().model_id == "m1"
    assert backend1.shutdown_called is False

    await reload_task
    assert reg.state() == RegistryState.READY
    assert reg.current_descriptor().model_id == "m2"
    assert backend1.shutdown_called is True

    await reg.shutdown()


@pytest.mark.asyncio
async def test_reload_worker_preserves_old_effective_on_failure():
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2", fail_init=True)
    next_backend = [backend1, backend2]

    def factory(config):
        return next_backend.pop(0)

    cfg1 = {"stt": {"backend": "fake", "fake": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "fake", "fake": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=factory)

    await reg.start()
    await asyncio.sleep(0.01)
    assert reg.current_descriptor().model_id == "m1"

    await reg._reload_worker(cfg2, compute_signature(cfg2))

    assert reg.is_ready() is True
    assert reg.current_descriptor().model_id == "m1"
    assert reg.state() == RegistryState.ERROR
    assert "init boom" in (reg.last_error() or "")
    assert backend1.shutdown_called is False

    await reg.shutdown()


@pytest.mark.asyncio
async def test_reload_worker_shuts_down_backend_if_describe_fails():
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2", fail_describe=True)
    next_backend = [backend1, backend2]

    def factory(config):
        return next_backend.pop(0)

    cfg1 = {"stt": {"backend": "fake", "fake": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "fake", "fake": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=factory)

    await reg.start()
    await asyncio.sleep(0.01)

    await reg._reload_worker(cfg2, compute_signature(cfg2))

    assert reg.is_ready() is True
    assert reg.current_descriptor().model_id == "m1"
    assert reg.state() == RegistryState.ERROR
    assert backend2.shutdown_called is True

    await reg.shutdown()
