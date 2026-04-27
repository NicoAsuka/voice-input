from __future__ import annotations

import asyncio
import enum
import hashlib
import logging
import json
from dataclasses import dataclass
from typing import Any, Callable


log = logging.getLogger(__name__)


class RegistryState(enum.Enum):
    LOADING = "loading"
    READY = "ready"
    RELOADING = "reloading"
    ERROR = "error"


def compute_signature(config: dict) -> str:
    """Compute a stable fingerprint for STT-relevant config only."""
    if not isinstance(config, dict):
        config = {}
    stt = config.get("stt")
    if not isinstance(stt, dict):
        stt = {}
    relevant = {
        "backend": stt.get("backend"),
        "sherpa": stt.get("sherpa"),
        "volcengine": stt.get("volcengine"),
        "google": stt.get("google"),
        "openai": stt.get("openai"),
    }
    serialized = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


StateListener = Callable[[RegistryState, str | None], None]


@dataclass(slots=True)
class _Effective:
    backend: Any
    descriptor: Any
    signature: str


class BackendRegistry:
    """Own the effective backend and reload it when STT config changes."""

    def __init__(self, config: dict, factory: Callable[[dict], Any]) -> None:
        self._config = config
        self._factory = factory
        self._effective: _Effective | None = None
        self._target_signature: str | None = None
        self._reload_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._listeners: list[StateListener] = []
        self._state = RegistryState.LOADING
        self._last_error: str | None = None
        self._generation = 0
        self._started = False
        self._closed = False

    def state(self) -> RegistryState:
        return self._state

    def last_error(self) -> str | None:
        return self._last_error

    def is_ready(self) -> bool:
        return self._effective is not None

    def current_descriptor(self) -> Any | None:
        return self._effective.descriptor if self._effective else None

    def add_state_listener(self, callback: StateListener) -> None:
        self._listeners.append(callback)

    def _set_state(
        self, state: RegistryState, error: str | None = None, *, force: bool = False
    ) -> list[StateListener]:
        if not force and self._state == state and self._last_error == error:
            return []
        self._state = state
        self._last_error = error
        return list(self._listeners)

    def _notify_listeners(
        self, listeners: list[StateListener], state: RegistryState, error: str | None
    ) -> None:
        for listener in listeners:
            try:
                listener(state, error)
            except Exception:
                log.exception("state listener raised")

    def create_session(self, language: str) -> Any:
        if self._effective is None:
            raise RuntimeError("Backend is not ready")
        return self._effective.backend.create_session(language)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._closed = False
        signature = compute_signature(self._config)
        self._target_signature = signature
        self._generation += 1
        generation = self._generation
        listeners = self._set_state(RegistryState.LOADING, force=True)
        self._reload_task = asyncio.create_task(
            self._reload_worker(self._config, signature, generation)
        )
        if not self._closed and generation == self._generation:
            self._notify_listeners(listeners, RegistryState.LOADING, None)

    async def synchronize(self, config: dict) -> None:
        new_signature = compute_signature(config)
        self._config = config
        if new_signature == self._target_signature:
            return

        previous_task = self._reload_task
        if previous_task is not None and not previous_task.done():
            previous_task.cancel()

        async with self._lock:
            self._target_signature = new_signature
            self._generation += 1
            generation = self._generation
            state = (
                RegistryState.RELOADING
                if self._effective is not None
                else RegistryState.LOADING
            )
            listeners = self._set_state(state)
            self._reload_task = asyncio.create_task(
                self._reload_worker(config, new_signature, generation)
            )

        if not self._closed and generation == self._generation:
            self._notify_listeners(listeners, state, None)

        if previous_task is not None and previous_task is not self._reload_task:
            try:
                await previous_task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("previous reload task failed")

    async def _reload_worker(
        self, config: dict, signature: str, generation: int
    ) -> None:
        backend: Any | None = None
        try:
            log.info("reload worker started for signature=%s", signature[:12])
            backend = self._factory(config)
            await backend.initialize()
            descriptor = backend.describe()
        except asyncio.CancelledError:
            if backend is not None:
                try:
                    await backend.shutdown()
                except Exception:
                    log.exception("cancelled backend shutdown failed")
            raise
        except Exception as exc:
            if backend is not None:
                try:
                    await backend.shutdown()
                except Exception:
                    log.exception("failed backend shutdown during error cleanup")
            if self._closed:
                return
            async with self._lock:
                if self._closed or generation != self._generation:
                    return
                log.exception("backend initialize failed")
                listeners = self._set_state(RegistryState.ERROR, error=str(exc))
            if not self._closed and generation == self._generation:
                self._notify_listeners(listeners, RegistryState.ERROR, str(exc))
            return

        if self._closed or generation != self._generation:
            try:
                await backend.shutdown()
            except Exception:
                log.exception("stale backend shutdown failed")
            return

        async with self._lock:
            if self._closed or generation != self._generation:
                try:
                    await backend.shutdown()
                except Exception:
                    log.exception("stale backend shutdown failed")
                return
            previous = self._effective
            self._effective = _Effective(
                backend=backend,
                descriptor=descriptor,
                signature=signature,
            )
            listeners = self._set_state(RegistryState.READY)

        if not self._closed and generation == self._generation:
            self._notify_listeners(listeners, RegistryState.READY, None)

        if previous is not None:
            try:
                await previous.backend.shutdown()
            except Exception:
                log.exception("old backend shutdown failed")

        if self._reload_task is not None and self._reload_task.done():
            self._reload_task = None

    async def shutdown(self) -> None:
        self._closed = True
        self._generation += 1
        task = self._reload_task
        self._reload_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("reload task failed during shutdown")
        async with self._lock:
            effective = self._effective
            self._effective = None
        if effective is not None:
            try:
                await effective.backend.shutdown()
            except Exception:
                log.exception("shutdown failed")
