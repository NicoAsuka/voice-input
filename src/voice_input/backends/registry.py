from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Callable

from voice_input.backends.base import BackendDescriptor, Session, TranscriptionBackend

log = logging.getLogger(__name__)


class RegistryState(enum.Enum):
    LOADING = "loading"
    READY = "ready"
    RELOADING = "reloading"
    ERROR = "error"


def compute_signature(config: dict) -> str:
    """Compute a stable fingerprint for STT-relevant config only."""
    stt = config.get("stt") or {}
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
    backend: TranscriptionBackend
    descriptor: BackendDescriptor
    signature: str


class BackendRegistry:
    """Manage STT backend lifecycle with background initialization."""

    def __init__(
        self,
        config: dict,
        factory: Callable[[dict], TranscriptionBackend],
    ) -> None:
        self._config = config
        self._factory = factory
        self._effective: _Effective | None = None
        self._target_signature: str | None = None
        self._reload_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._listeners: list[StateListener] = []
        self._state = RegistryState.LOADING
        self._last_error: str | None = None

    def state(self) -> RegistryState:
        return self._state

    def last_error(self) -> str | None:
        return self._last_error

    def is_ready(self) -> bool:
        return self._effective is not None

    def current_descriptor(self) -> BackendDescriptor | None:
        if self._effective is None:
            return None
        return self._effective.descriptor

    def add_state_listener(self, callback: StateListener) -> None:
        self._listeners.append(callback)

    def _set_state(self, state: RegistryState, error: str | None = None) -> None:
        self._state = state
        self._last_error = error
        for callback in list(self._listeners):
            try:
                callback(state, error)
            except Exception:
                log.exception("state listener raised")

    def create_session(self, language: str) -> Session:
        if self._effective is None:
            raise RuntimeError("Backend is not ready")
        return self._effective.backend.create_session(language)

    async def start(self) -> None:
        """Trigger initial backend load in the background."""
        signature = compute_signature(self._config)
        if self._reload_task is not None and not self._reload_task.done():
            return

        self._target_signature = signature
        self._set_state(RegistryState.LOADING)
        self._reload_task = asyncio.create_task(
            self._reload_worker(self._config, signature)
        )

    async def _reload_worker(self, config: dict, signature: str) -> None:
        """Load a backend, then swap it in atomically on success."""
        log.info("reload worker started for signature=%s", signature[:12])
        new_backend: TranscriptionBackend | None = None
        try:
            new_backend = self._factory(config)
            await new_backend.initialize()
            descriptor = new_backend.describe()
        except asyncio.CancelledError:
            log.info("reload cancelled")
            if new_backend is not None:
                try:
                    await new_backend.shutdown()
                except Exception:
                    log.exception("abandoned backend shutdown failed")
            raise
        except Exception as exc:
            log.exception("backend initialize failed")
            if new_backend is not None:
                try:
                    await new_backend.shutdown()
                except Exception:
                    log.exception("abandoned backend shutdown failed")
            self._set_state(RegistryState.ERROR, error=str(exc))
            return

        async with self._lock:
            old = self._effective
            self._effective = _Effective(
                backend=new_backend,
                descriptor=descriptor,
                signature=signature,
            )

        if old is not None:
            try:
                await old.backend.shutdown()
            except Exception:
                log.exception("old backend shutdown failed")

        self._set_state(RegistryState.READY)
        log.info("backend ready: %s", descriptor.model_id)

    async def shutdown(self) -> None:
        if self._reload_task is not None and not self._reload_task.done():
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        async with self._lock:
            effective = self._effective
            self._effective = None

        if effective is not None:
            try:
                await effective.backend.shutdown()
            except Exception:
                log.exception("shutdown failed")
