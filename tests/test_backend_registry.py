from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from voice_input.backends.registry import (
    BackendRegistry,
    RegistryState,
    compute_signature,
)


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


def test_compute_signature_handles_null_stt():
    assert compute_signature({"stt": None}) == compute_signature({})


class FakeSession:
    def __init__(self, backend: "FakeBackend") -> None:
        self._backend = backend
        self.finished = False
        self.cancelled = False

    def push_audio(self, pcm_int16) -> None:
        self._backend.pushed_audio.append(pcm_int16)

    async def finish(self) -> str:
        if self._backend.finish_delay:
            await asyncio.sleep(self._backend.finish_delay)
        self.finished = True
        return self._backend.session_text

    def cancel(self) -> None:
        self.cancelled = True


class FakeBackend:
    def __init__(
        self,
        *,
        model_id: str = "m",
        init_delay: float = 0.0,
        fail_init: bool = False,
        session_text: str = "fake-text",
        finish_delay: float = 0.0,
        init_gate: asyncio.Event | None = None,
    ) -> None:
        self.model_id = model_id
        self.init_delay = init_delay
        self.fail_init = fail_init
        self.session_text = session_text
        self.finish_delay = finish_delay
        self.init_gate = init_gate
        self.initialize_started = asyncio.Event()
        self.initialize_finished = asyncio.Event()
        self.shutdown_called = False
        self.init_call_count = 0
        self.describe_call_count = 0
        self.create_session_call_count = 0
        self.pushed_audio: list[object] = []

    async def initialize(self) -> None:
        self.init_call_count += 1
        self.initialize_started.set()
        try:
            if self.init_delay:
                await asyncio.sleep(self.init_delay)
            if self.init_gate is not None:
                await self.init_gate.wait()
            if self.fail_init:
                raise RuntimeError("init boom")
        finally:
            self.initialize_finished.set()

    def describe(self):
        self.describe_call_count += 1
        return SimpleNamespace(
            backend_id="fake",
            model_id=self.model_id,
            capabilities=SimpleNamespace(supports_vad=False),
        )

    def create_session(self, language: str) -> FakeSession:
        self.create_session_call_count += 1
        self.last_language = language
        return FakeSession(self)

    async def shutdown(self) -> None:
        self.shutdown_called = True


def make_factory(backends: dict[str, FakeBackend]):
    def factory(config: dict) -> FakeBackend:
        stt = config.get("stt", {}) if isinstance(config, dict) else {}
        sherpa_cfg = stt.get("sherpa") or {}
        key = sherpa_cfg.get("model_id") or stt.get("backend")
        if key not in backends:
            raise KeyError(f"unknown backend key: {key!r}")
        return backends[key]

    return factory


@pytest.mark.asyncio
async def test_start_returns_immediately_while_loading():
    slow_backend = FakeBackend(init_delay=0.2)
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory({"fake": slow_backend}))

    t0 = asyncio.get_running_loop().time()
    await reg.start()
    elapsed = asyncio.get_running_loop().time() - t0
    assert elapsed < 0.1, f"start() should be non-blocking, took {elapsed}s"

    assert reg.is_ready() is False
    assert reg.state() == RegistryState.LOADING

    await asyncio.wait_for(slow_backend.initialize_finished.wait(), timeout=1)
    assert reg.is_ready() is True
    assert reg.state() == RegistryState.READY

    await reg.shutdown()


@pytest.mark.asyncio
async def test_create_session_raises_when_not_ready():
    slow_backend = FakeBackend(init_delay=10.0)
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory({"fake": slow_backend}))
    await reg.start()
    with pytest.raises(RuntimeError, match="not ready"):
        reg.create_session("zh")
    await reg.shutdown()


@pytest.mark.asyncio
async def test_create_session_works_when_ready():
    backend = FakeBackend()
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory({"fake": backend}))
    await reg.start()
    await asyncio.wait_for(backend.initialize_finished.wait(), timeout=1)
    assert reg.is_ready()
    session = reg.create_session("zh")
    assert isinstance(session, FakeSession)
    assert backend.last_language == "zh"
    await reg.shutdown()


@pytest.mark.asyncio
async def test_reload_failure_keeps_no_effective_initially():
    bad = FakeBackend(fail_init=True)
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory({"fake": bad}))
    await reg.start()
    await asyncio.wait_for(bad.initialize_started.wait(), timeout=1)
    await asyncio.wait_for(bad.initialize_finished.wait(), timeout=1)
    assert reg.is_ready() is False
    assert reg.state() == RegistryState.ERROR
    assert "init boom" in (reg.last_error() or "")
    await reg.shutdown()


@pytest.mark.asyncio
async def test_current_descriptor_returns_none_when_not_ready():
    slow = FakeBackend(init_delay=10.0)
    cfg = {"stt": {"backend": "fake"}}
    reg = BackendRegistry(cfg, factory=make_factory({"fake": slow}))
    await reg.start()
    assert reg.current_descriptor() is None
    await reg.shutdown()


@pytest.mark.asyncio
async def test_synchronize_uses_config_snapshot():
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2")
    seen: list[str] = []

    def factory(config: dict) -> FakeBackend:
        model_id = config["stt"]["sherpa"]["model_id"]
        seen.append(model_id)
        return {"m1": backend1, "m2": backend2}[model_id]

    cfg1 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=factory)

    await reg.start()
    await asyncio.wait_for(backend1.initialize_finished.wait(), timeout=1)

    await reg.synchronize(cfg2)
    cfg2["stt"]["sherpa"]["model_id"] = "mutated"
    await asyncio.wait_for(backend2.initialize_finished.wait(), timeout=1)

    assert seen == ["m1", "m2"]
    assert reg.current_descriptor().model_id == "m2"
    await reg.shutdown()


@pytest.mark.asyncio
async def test_synchronize_no_op_when_signature_unchanged():
    backend = FakeBackend(model_id="m1")
    cfg = {"stt": {"backend": "fake", "sherpa": {"model_id": "m1"}}}
    reg = BackendRegistry(cfg, factory=make_factory({"m1": backend, "fake": backend}))
    await reg.start()
    await asyncio.wait_for(backend.initialize_finished.wait(), timeout=1)
    assert backend.init_call_count == 1

    await reg.synchronize(cfg)
    await asyncio.sleep(0.05)

    assert backend.init_call_count == 1
    assert reg.current_descriptor().model_id == "m1"
    assert reg.state() == RegistryState.READY
    await reg.shutdown()


@pytest.mark.asyncio
async def test_synchronize_triggers_reload_when_signature_changes():
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2")
    cfg1 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=make_factory({"m1": backend1, "m2": backend2}))
    await reg.start()
    await asyncio.wait_for(backend1.initialize_finished.wait(), timeout=1)
    assert reg.current_descriptor().model_id == "m1"

    await reg.synchronize(cfg2)
    await asyncio.wait_for(backend2.initialize_finished.wait(), timeout=1)

    assert reg.current_descriptor().model_id == "m2"
    assert backend1.shutdown_called is True
    await reg.shutdown()


@pytest.mark.asyncio
async def test_reload_keeps_old_effective_during_reload():
    backend1 = FakeBackend(model_id="m1")
    gate = asyncio.Event()
    backend2 = FakeBackend(model_id="m2", init_gate=gate)
    cfg1 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=make_factory({"m1": backend1, "m2": backend2}))
    await reg.start()
    await asyncio.wait_for(backend1.initialize_finished.wait(), timeout=1)

    await reg.synchronize(cfg2)
    await asyncio.wait_for(backend2.initialize_started.wait(), timeout=1)

    assert reg.is_ready() is True
    assert reg.current_descriptor().model_id == "m1"
    assert reg.state() == RegistryState.RELOADING

    gate.set()
    await asyncio.wait_for(backend2.initialize_finished.wait(), timeout=1)
    assert reg.current_descriptor().model_id == "m2"
    await reg.shutdown()


@pytest.mark.asyncio
async def test_reload_failure_keeps_old_effective():
    backend1 = FakeBackend(model_id="m1")
    backend2 = FakeBackend(model_id="m2", fail_init=True)
    cfg1 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=make_factory({"m1": backend1, "m2": backend2}))
    await reg.start()
    await asyncio.wait_for(backend1.initialize_finished.wait(), timeout=1)

    await reg.synchronize(cfg2)
    await asyncio.wait_for(backend2.initialize_started.wait(), timeout=1)
    await asyncio.wait_for(backend2.initialize_finished.wait(), timeout=1)

    assert reg.is_ready() is True
    assert reg.current_descriptor().model_id == "m1"
    assert reg.state() == RegistryState.ERROR
    assert "init boom" in (reg.last_error() or "")
    assert backend1.shutdown_called is False
    await reg.shutdown()


@pytest.mark.asyncio
async def test_listeners_notified_on_state_change():
    backend1 = FakeBackend(model_id="m1")
    gate = asyncio.Event()
    backend2 = FakeBackend(model_id="m2", init_gate=gate)
    cfg1 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=make_factory({"m1": backend1, "m2": backend2}))

    events: list[tuple[RegistryState, str | None]] = []
    reg.add_state_listener(lambda state, err: events.append((state, err)))

    await reg.start()
    await asyncio.wait_for(backend1.initialize_finished.wait(), timeout=1)

    await reg.synchronize(cfg2)
    await asyncio.wait_for(backend2.initialize_started.wait(), timeout=1)
    gate.set()
    await asyncio.wait_for(backend2.initialize_finished.wait(), timeout=1)

    states = [state for state, _ in events]
    assert RegistryState.LOADING in states
    assert RegistryState.RELOADING in states
    assert RegistryState.READY in states
    await reg.shutdown()


@pytest.mark.asyncio
async def test_concurrent_reload_cancels_previous():
    backend1 = FakeBackend(model_id="m1")
    gate2 = asyncio.Event()
    backend2 = FakeBackend(model_id="m2", init_gate=gate2)
    backend3 = FakeBackend(model_id="m3")
    cfg1 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m2"}}}
    cfg3 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m3"}}}
    reg = BackendRegistry(
        cfg1,
        factory=make_factory({"m1": backend1, "m2": backend2, "m3": backend3}),
    )
    await reg.start()
    await asyncio.wait_for(backend1.initialize_finished.wait(), timeout=1)

    await reg.synchronize(cfg2)
    await asyncio.wait_for(backend2.initialize_started.wait(), timeout=1)

    await reg.synchronize(cfg3)
    await asyncio.wait_for(backend3.initialize_finished.wait(), timeout=1)
    gate2.set()

    assert reg.current_descriptor().model_id == "m3"
    await reg.shutdown()


@pytest.mark.asyncio
async def test_session_keeps_backend_alive_during_reload():
    backend1 = FakeBackend(model_id="m1", finish_delay=0.15)
    gate = asyncio.Event()
    backend2 = FakeBackend(model_id="m2", init_gate=gate)
    cfg1 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m1"}}}
    cfg2 = {"stt": {"backend": "fake", "sherpa": {"model_id": "m2"}}}
    reg = BackendRegistry(cfg1, factory=make_factory({"m1": backend1, "m2": backend2}))
    await reg.start()
    await asyncio.wait_for(backend1.initialize_finished.wait(), timeout=1)

    session = reg.create_session("zh")
    assert isinstance(session, FakeSession)

    finish_task = asyncio.create_task(session.finish())
    await reg.synchronize(cfg2)
    await asyncio.wait_for(backend2.initialize_started.wait(), timeout=1)
    gate.set()

    assert await asyncio.wait_for(finish_task, timeout=1) == "fake-text"
    await asyncio.wait_for(backend2.initialize_finished.wait(), timeout=1)
    assert reg.current_descriptor().model_id == "m2"
    await reg.shutdown()
