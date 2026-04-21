# SenseVoice Small Local Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SenseVoice Small as a switchable local STT engine alongside faster-whisper, using a strategy pattern inside the local backend.

**Architecture:** Restructure `backends/local_whisper.py` into a `backends/local/` package with a `LocalBackend` that delegates to engine implementations (`WhisperEngine`, `SenseVoiceEngine`) selected by config. Non-streaming engines trigger a new `TRANSCRIBING` app state where audio is transcribed after recording ends.

**Tech Stack:** funasr, modelscope (optional deps), PyQt6, numpy, faster-whisper (optional dep)

---

### Task 1: Create LocalEngine Protocol and engine base

**Files:**
- Create: `src/voice_input/backends/local/__init__.py`
- Create: `src/voice_input/backends/local/engine.py`
- Create: `tests/test_local_engine.py`

- [ ] **Step 1: Write the test for LocalEngine protocol conformance**

```python
# tests/test_local_engine.py
import numpy as np
from pathlib import Path
from voice_input.backends.local.engine import LocalEngine


def test_protocol_conformance_with_complete_class():
    """A class implementing all protocol methods is recognized."""
    class FakeEngine:
        def load_model(self, model_name: str, device: str, cache_dir: Path) -> None:
            pass

        def transcribe(self, audio_f32: np.ndarray, language: str) -> str:
            return ""

        def is_streaming(self) -> bool:
            return False

    engine: LocalEngine = FakeEngine()
    assert engine.is_streaming() is False
    assert engine.transcribe(np.zeros(100, dtype=np.float32), "zh") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_local_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voice_input.backends.local'`

- [ ] **Step 3: Create the engine protocol**

```python
# src/voice_input/backends/local/engine.py
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LocalEngine(Protocol):
    """Protocol for local STT engine implementations."""

    def load_model(self, model_name: str, device: str, cache_dir: Path) -> None:
        """Load model into memory."""
        ...

    def transcribe(self, audio_f32: np.ndarray, language: str) -> str:
        """Transcribe float32 audio array, return text."""
        ...

    def is_streaming(self) -> bool:
        """Whether this engine supports real-time incremental transcription."""
        ...
```

```python
# src/voice_input/backends/local/__init__.py
"""Local STT backend with switchable engine (whisper / sensevoice)."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_local_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/local/__init__.py src/voice_input/backends/local/engine.py tests/test_local_engine.py
git commit -m "feat: add LocalEngine protocol for local STT engine abstraction"
```

---

### Task 2: Migrate WhisperEngine from local_whisper.py

**Files:**
- Create: `src/voice_input/backends/local/whisper_engine.py`
- Create: `tests/test_whisper_engine.py`

- [ ] **Step 1: Write failing tests for WhisperEngine**

```python
# tests/test_whisper_engine.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_whisper_engine_is_streaming():
    from voice_input.backends.local.whisper_engine import WhisperEngine
    engine = WhisperEngine()
    assert engine.is_streaming() is True


def test_whisper_engine_load_model_cpu():
    from voice_input.backends.local.whisper_engine import WhisperEngine
    engine = WhisperEngine()
    mock_model = MagicMock()
    with patch("voice_input.backends.local.whisper_engine.WhisperModel", return_value=mock_model):
        engine.load_model("tiny", "cpu", Path("/tmp/models"))
    assert engine._model is mock_model


def test_whisper_engine_transcribe():
    from voice_input.backends.local.whisper_engine import WhisperEngine
    engine = WhisperEngine()
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)
    engine._model = mock_model

    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, "zh")
    assert result == "hello world"


def test_whisper_engine_transcribe_no_model_returns_empty():
    from voice_input.backends.local.whisper_engine import WhisperEngine
    engine = WhisperEngine()
    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, "zh")
    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_whisper_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement WhisperEngine**

```python
# src/voice_input/backends/local/whisper_engine.py
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]


class WhisperEngine:
    """Local faster-whisper engine implementation."""

    def __init__(self) -> None:
        self._model = None

    def is_streaming(self) -> bool:
        return True

    def load_model(self, model_name: str, device: str, cache_dir: Path) -> None:
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install with: pip install voice-input[whisper]"
            )
        compute_type = "int8"
        actual_device = "cpu"
        if device in ("auto", "cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    actual_device = "cuda"
                    compute_type = "float16"
            except ImportError:
                pass

        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "Loading whisper model=%s device=%s compute_type=%s",
            model_name, actual_device, compute_type,
        )
        self._model = WhisperModel(
            model_name,
            device=actual_device,
            compute_type=compute_type,
            download_root=str(cache_dir),
        )
        log.info("Whisper model loaded and ready")

    def transcribe(self, audio_f32: np.ndarray, language: str) -> str:
        if self._model is None:
            return ""
        try:
            segments, _ = self._model.transcribe(
                audio_f32,
                language=language,
                beam_size=5,
                vad_filter=False,
            )
            text = "".join(seg.text for seg in segments)
            return text.strip()
        except Exception as e:
            log.error("Whisper transcription error: %s", e)
            return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_whisper_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/local/whisper_engine.py tests/test_whisper_engine.py
git commit -m "feat: add WhisperEngine extracted from LocalWhisperBackend"
```

---

### Task 3: Implement SenseVoiceEngine

**Files:**
- Create: `src/voice_input/backends/local/sensevoice_engine.py`
- Create: `tests/test_sensevoice_engine.py`

- [ ] **Step 1: Write failing tests for SenseVoiceEngine**

```python
# tests/test_sensevoice_engine.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_sensevoice_engine_is_streaming():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine
    engine = SenseVoiceEngine()
    assert engine.is_streaming() is False


def test_sensevoice_engine_load_model():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine
    engine = SenseVoiceEngine()
    mock_model = MagicMock()
    with patch("voice_input.backends.local.sensevoice_engine.AutoModel", mock_model):
        engine.load_model("iic/SenseVoiceSmall", "cpu", Path("/tmp/models"))
    mock_model.assert_called_once_with(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="cpu",
        cache_dir="/tmp/models",
    )


def test_sensevoice_engine_load_model_default_name():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine
    engine = SenseVoiceEngine()
    mock_model = MagicMock()
    with patch("voice_input.backends.local.sensevoice_engine.AutoModel", mock_model):
        engine.load_model("", "cpu", Path("/tmp/models"))
    mock_model.assert_called_once_with(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        device="cpu",
        cache_dir="/tmp/models",
    )


def test_sensevoice_engine_load_model_missing_funasr():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine
    engine = SenseVoiceEngine()
    with patch("voice_input.backends.local.sensevoice_engine.AutoModel", None):
        with pytest.raises(RuntimeError, match="funasr"):
            engine.load_model("iic/SenseVoiceSmall", "cpu", Path("/tmp/models"))


def test_sensevoice_engine_transcribe():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine
    engine = SenseVoiceEngine()
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"text": "你好世界"}]
    engine._model = mock_model

    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, "zh")
    assert result == "你好世界"
    mock_model.generate.assert_called_once_with(
        input=audio,
        language="zh",
        use_itn=True,
    )


def test_sensevoice_engine_transcribe_empty_result():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine
    engine = SenseVoiceEngine()
    mock_model = MagicMock()
    mock_model.generate.return_value = []
    engine._model = mock_model

    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, "zh")
    assert result == ""


def test_sensevoice_engine_transcribe_no_model():
    from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine
    engine = SenseVoiceEngine()
    audio = np.zeros(16000, dtype=np.float32)
    result = engine.transcribe(audio, "zh")
    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sensevoice_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement SenseVoiceEngine**

```python
# src/voice_input/backends/local/sensevoice_engine.py
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:
    from funasr import AutoModel
except ImportError:
    AutoModel = None  # type: ignore[assignment,misc]

DEFAULT_MODEL = "iic/SenseVoiceSmall"


class SenseVoiceEngine:
    """Local SenseVoice Small engine using funasr."""

    def __init__(self) -> None:
        self._model = None

    def is_streaming(self) -> bool:
        return False

    def load_model(self, model_name: str, device: str, cache_dir: Path) -> None:
        if AutoModel is None:
            raise RuntimeError(
                "funasr is not installed. "
                "Install with: pip install voice-input[sensevoice]"
            )
        actual_device = self._resolve_device(device)
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved_name = model_name or DEFAULT_MODEL
        log.info(
            "Loading SenseVoice model=%s device=%s",
            resolved_name, actual_device,
        )
        self._model = AutoModel(
            model=resolved_name,
            trust_remote_code=True,
            device=actual_device,
            cache_dir=str(cache_dir),
        )
        log.info("SenseVoice model loaded and ready")

    def transcribe(self, audio_f32: np.ndarray, language: str) -> str:
        if self._model is None:
            return ""
        try:
            result = self._model.generate(
                input=audio_f32,
                language=language,
                use_itn=True,
            )
            if result and len(result) > 0:
                return result[0].get("text", "").strip()
            return ""
        except Exception as e:
            log.error("SenseVoice transcription error: %s", e)
            return ""

    def _resolve_device(self, device: str) -> str:
        if device in ("auto", "cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda:0"
            except ImportError:
                pass
        return "cpu"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sensevoice_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/local/sensevoice_engine.py tests/test_sensevoice_engine.py
git commit -m "feat: add SenseVoiceEngine using funasr for local transcription"
```

---

### Task 4: Implement LocalBackend (strategy host)

**Files:**
- Modify: `src/voice_input/backends/local/__init__.py`
- Create: `tests/test_local_backend.py`

- [ ] **Step 1: Write failing tests for LocalBackend**

```python
# tests/test_local_backend.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_local_backend_initializes_whisper_engine():
    from voice_input.backends.local import LocalBackend
    config = {
        "stt": {"backend": "local", "local": {
            "engine": "whisper", "model": "tiny",
            "language": "zh", "device": "cpu",
        }},
    }
    backend = LocalBackend(config)
    mock_engine = MagicMock()
    with patch("voice_input.backends.local.WhisperEngine", return_value=mock_engine):
        await backend.initialize()
    mock_engine.load_model.assert_called_once()
    assert backend.is_streaming() == mock_engine.is_streaming()


@pytest.mark.asyncio
async def test_local_backend_initializes_sensevoice_engine():
    from voice_input.backends.local import LocalBackend
    config = {
        "stt": {"backend": "local", "local": {
            "engine": "sensevoice", "model": "iic/SenseVoiceSmall",
            "language": "zh", "device": "cpu",
        }},
    }
    backend = LocalBackend(config)
    mock_engine = MagicMock()
    with patch("voice_input.backends.local.SenseVoiceEngine", return_value=mock_engine):
        await backend.initialize()
    mock_engine.load_model.assert_called_once()


@pytest.mark.asyncio
async def test_local_backend_unknown_engine_raises():
    from voice_input.backends.local import LocalBackend
    config = {
        "stt": {"backend": "local", "local": {
            "engine": "unknown", "model": "x",
            "language": "zh", "device": "cpu",
        }},
    }
    backend = LocalBackend(config)
    with pytest.raises(ValueError, match="Unknown local engine"):
        await backend.initialize()


@pytest.mark.asyncio
async def test_local_backend_transcribe_delegates():
    from voice_input.backends.local import LocalBackend
    config = {
        "stt": {"backend": "local", "local": {
            "engine": "whisper", "model": "tiny",
            "language": "zh", "device": "cpu",
        }},
    }
    backend = LocalBackend(config)
    mock_engine = MagicMock()
    mock_engine.transcribe.return_value = "hello"
    backend._engine = mock_engine

    audio = np.zeros(16000, dtype=np.int16)
    result = await backend.transcribe(audio, "zh")
    assert result == "hello"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_local_backend.py -v`
Expected: FAIL — `ImportError: cannot import name 'LocalBackend'`

- [ ] **Step 3: Implement LocalBackend**

```python
# src/voice_input/backends/local/__init__.py
"""Local STT backend with switchable engine (whisper / sensevoice)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from voice_input.backends.base import TranscriptionBackend
from voice_input.config import AppConfig, xdg_cache_dir

log = logging.getLogger(__name__)

# Minimum samples to attempt transcription (0.1s at 16kHz)
MIN_SAMPLES = 1600


class LocalBackend(TranscriptionBackend):
    """Local transcription backend that delegates to a switchable engine."""

    def __init__(self, config: AppConfig) -> None:
        local_cfg = config.get("stt", {}).get("local", {})
        self._engine_name = local_cfg.get("engine", "whisper")
        self._model_name = local_cfg.get("model", "medium")
        self._device = local_cfg.get("device", "auto")
        self._engine = None

    def is_streaming(self) -> bool:
        if self._engine is None:
            # Default assumption before initialization
            return self._engine_name == "whisper"
        return self._engine.is_streaming()

    async def initialize(self) -> None:
        if self._engine_name == "whisper":
            from voice_input.backends.local.whisper_engine import WhisperEngine
            self._engine = WhisperEngine()
        elif self._engine_name == "sensevoice":
            from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine
            self._engine = SenseVoiceEngine()
        else:
            raise ValueError(f"Unknown local engine: {self._engine_name}")

        cache_dir = xdg_cache_dir() / "models"
        self._engine.load_model(self._model_name, self._device, cache_dir)

    async def transcribe(self, audio_data: np.ndarray, language: str) -> str:
        if self._engine is None or len(audio_data) < MIN_SAMPLES:
            return ""
        audio_f32 = audio_data.astype(np.float32) / 32768.0
        return self._engine.transcribe(audio_f32, language)

    async def cleanup(self) -> None:
        self._engine = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_local_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/local/__init__.py tests/test_local_backend.py
git commit -m "feat: add LocalBackend with engine strategy delegation"
```

---

### Task 5: Update config.py — migrate whisper config to stt.local

**Files:**
- Modify: `src/voice_input/config.py:19-27`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing test for new config structure**

Add to `tests/test_config.py`:

```python
def test_default_config_has_stt_local():
    from voice_input.config import DEFAULT_CONFIG
    assert "stt" in DEFAULT_CONFIG
    assert "local" in DEFAULT_CONFIG["stt"]
    local = DEFAULT_CONFIG["stt"]["local"]
    assert local["engine"] == "whisper"
    assert local["model"] == "medium"
    assert local["language"] == "zh"
    assert local["device"] == "auto"


def test_default_config_no_whisper_section():
    """Old whisper section should be removed."""
    from voice_input.config import DEFAULT_CONFIG
    assert "whisper" not in DEFAULT_CONFIG
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py::test_default_config_has_stt_local tests/test_config.py::test_default_config_no_whisper_section -v`
Expected: FAIL

- [ ] **Step 3: Update DEFAULT_CONFIG in config.py**

In `src/voice_input/config.py`, replace the `"whisper"` section (lines 24-28) with the new `"stt"` section:

```python
DEFAULT_CONFIG: dict = {
    "hotkey": {
        "mode": "toggle",
        "key": "Meta+Space",
    },
    "stt": {
        "backend": "local",
        "local": {
            "engine": "whisper",
            "model": "medium",
            "language": "zh",
            "device": "auto",
        },
    },
    "llm": {
        "enabled": True,
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "audio": {
        "device": "default",
        "sample_rate": 16000,
    },
    "ui": {
        "overlay_margin_bottom": 80,
    },
    "inject": {
        "paste_method": "ctrl_v",
    },
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/config.py tests/test_config.py
git commit -m "refactor: migrate whisper config to stt.local section"
```

---

### Task 6: Update backends/__init__.py with factory function

**Files:**
- Modify: `src/voice_input/backends/__init__.py`
- Create: `tests/test_backend_factory.py`

- [ ] **Step 1: Write failing tests for create_backend factory**

```python
# tests/test_backend_factory.py
import pytest
from unittest.mock import patch, MagicMock


def test_create_backend_local():
    from voice_input.backends import create_backend
    config = {
        "stt": {"backend": "local", "local": {
            "engine": "whisper", "model": "tiny",
            "language": "zh", "device": "cpu",
        }},
    }
    backend = create_backend(config)
    from voice_input.backends.local import LocalBackend
    assert isinstance(backend, LocalBackend)


def test_create_backend_unknown_raises():
    from voice_input.backends import create_backend
    config = {"stt": {"backend": "nonexistent"}}
    with pytest.raises(ValueError, match="Unknown STT backend"):
        create_backend(config)


def test_create_backend_default_is_local():
    from voice_input.backends import create_backend
    config = {"stt": {"local": {
        "engine": "whisper", "model": "tiny",
        "language": "zh", "device": "cpu",
    }}}
    backend = create_backend(config)
    from voice_input.backends.local import LocalBackend
    assert isinstance(backend, LocalBackend)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backend_factory.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_backend'`

- [ ] **Step 3: Implement factory function**

```python
# src/voice_input/backends/__init__.py
"""STT backend abstraction and factory."""
from __future__ import annotations

from voice_input.backends.base import TranscriptionBackend
from voice_input.config import AppConfig


def create_backend(config: AppConfig) -> TranscriptionBackend:
    """Create the appropriate STT backend based on config."""
    backend_name = config.get("stt", {}).get("backend", "local")
    if backend_name == "local":
        from voice_input.backends.local import LocalBackend
        return LocalBackend(config)
    else:
        raise ValueError(f"Unknown STT backend: {backend_name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backend_factory.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/backends/__init__.py tests/test_backend_factory.py
git commit -m "feat: add create_backend factory function"
```

---

### Task 7: Add TRANSCRIBING state to AppState

**Files:**
- Modify: `src/voice_input/app.py:29-44`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write failing tests for TRANSCRIBING state**

Add to `tests/test_app.py`:

```python
def test_transcribing_state_exists():
    assert hasattr(AppState, "TRANSCRIBING")
    assert AppState.TRANSCRIBING.value == "Transcribing"


def test_transcribing_state_transitions():
    assert AppState.RECORDING.can_transition_to(AppState.TRANSCRIBING)
    assert AppState.TRANSCRIBING.can_transition_to(AppState.IDLE)
    assert AppState.TRANSCRIBING.can_transition_to(AppState.REFINING)
    assert not AppState.IDLE.can_transition_to(AppState.TRANSCRIBING)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app.py -v`
Expected: FAIL — `AttributeError: TRANSCRIBING`

- [ ] **Step 3: Add TRANSCRIBING state**

In `src/voice_input/app.py`, update the `AppState` enum and `_VALID_TRANSITIONS`:

```python
class AppState(enum.Enum):
    IDLE = "Idle"
    RECORDING = "Recording"
    TRANSCRIBING = "Transcribing"
    REFINING = "Refining"

    _transitions = None  # placeholder, set below

    def can_transition_to(self, target: AppState) -> bool:
        return target in _VALID_TRANSITIONS.get(self, set())


_VALID_TRANSITIONS = {
    AppState.IDLE: {AppState.RECORDING},
    AppState.RECORDING: {AppState.IDLE, AppState.TRANSCRIBING, AppState.REFINING},
    AppState.TRANSCRIBING: {AppState.IDLE, AppState.REFINING},
    AppState.REFINING: {AppState.IDLE},
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/app.py tests/test_app.py
git commit -m "feat: add TRANSCRIBING state for non-streaming engines"
```

---

### Task 8: Update AppController to use backend and handle non-streaming

**Files:**
- Modify: `src/voice_input/app.py:47-298`
- Modify: `src/voice_input/whisper_worker.py`

This is the integration task. `AppController` switches from directly constructing `WhisperWorker` with config values to using `create_backend()` and passing the backend to `WhisperWorker`. The stop-recording flow branches on `is_streaming()`.

- [ ] **Step 1: Update WhisperWorker to accept a backend**

In `src/voice_input/whisper_worker.py`, modify `__init__` and `run` to use a backend instead of direct faster-whisper. Add `get_buffer()` method:

```python
# src/voice_input/whisper_worker.py
from __future__ import annotations

import asyncio
import logging
import queue

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

MAX_BUFFER_SECONDS = 30
SAMPLE_RATE = 16000
MAX_BUFFER_SAMPLES = MAX_BUFFER_SECONDS * SAMPLE_RATE


class WhisperWorker(QThread):
    """Runs transcription in a background thread.

    Polls the audio queue, accumulates into a buffer (capped at 30s).
    For streaming engines: runs transcription each poll cycle.
    For non-streaming engines: only buffers, transcription happens externally.
    """

    transcription_updated = pyqtSignal(str)
    model_ready = pyqtSignal()
    error_occurred = pyqtSignal(str)

    POLL_INTERVAL_MS = 500

    def __init__(
        self,
        whisper_queue: queue.Queue,
        backend,
        language: str = "zh",
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.whisper_queue = whisper_queue
        self._backend = backend
        self.language = language
        self.audio_buffer = np.array([], dtype=np.int16)
        self._running = False
        self._active = False

    def set_active(self, active: bool) -> None:
        self._active = active

    def reset(self) -> None:
        self.audio_buffer = np.array([], dtype=np.int16)
        while True:
            try:
                self.whisper_queue.get_nowait()
            except queue.Empty:
                break

    def get_buffer(self) -> np.ndarray:
        """Return a copy of the current audio buffer."""
        return self.audio_buffer.copy()

    def _drain_and_accumulate(self) -> None:
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self.whisper_queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return
        new_data = np.concatenate(chunks)
        self.audio_buffer = np.concatenate([self.audio_buffer, new_data])
        if len(self.audio_buffer) > MAX_BUFFER_SAMPLES:
            self.audio_buffer = self.audio_buffer[-MAX_BUFFER_SAMPLES:]

    def run(self) -> None:
        try:
            asyncio.run(self._backend.initialize())
        except Exception as e:
            log.error("Backend initialization failed: %s", e)
            self.error_occurred.emit(f"Model load failed: {e}")
            return
        self.model_ready.emit()
        self._running = True
        streaming = self._backend.is_streaming()
        while self._running:
            self.msleep(self.POLL_INTERVAL_MS)
            if not self._active:
                continue
            self._drain_and_accumulate()
            if streaming and len(self.audio_buffer) >= 1600:
                try:
                    text = asyncio.run(
                        self._backend.transcribe(self.audio_buffer, self.language)
                    )
                    if text:
                        log.debug("Transcription: %s", text[:80])
                        self.transcription_updated.emit(text)
                except Exception as e:
                    log.error("Transcription error: %s", e)

    def stop(self) -> None:
        self._running = False
```

- [ ] **Step 2: Update AppController to use create_backend**

In `src/voice_input/app.py`, update imports and `__init__`:

Add import at top:
```python
from voice_input.backends import create_backend
```

In `AppController.__init__`, replace `WhisperWorker` construction:

```python
# Replace:
#   self._whisper = WhisperWorker(
#       whisper_queue=self._whisper_queue,
#       model_name=config["whisper"]["model"],
#       language=config["whisper"]["language"],
#       device=config["whisper"]["device"],
#   )

# With:
self._backend = create_backend(config)
self._language = config.get("stt", {}).get("local", {}).get("language", "zh")
self._whisper = WhisperWorker(
    whisper_queue=self._whisper_queue,
    backend=self._backend,
    language=self._language,
)
```

- [ ] **Step 3: Update _on_stop_recording for non-streaming**

In `src/voice_input/app.py`, replace `_on_stop_recording`:

```python
@pyqtSlot()
def _on_stop_recording(self) -> None:
    if self._state != AppState.RECORDING:
        return
    self._audio.stop()
    self._viz_timer.stop()
    self._whisper.set_active(False)

    if self._backend.is_streaming():
        text = self._last_transcription
        self._finish_transcription(text)
    else:
        buffer = self._whisper.get_buffer()
        if len(buffer) < 1600:
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return
        self._set_state(AppState.TRANSCRIBING)
        self._overlay.update_text("识别中...")
        asyncio.ensure_future(self._transcribe_and_finish(buffer))

async def _transcribe_and_finish(self, buffer: np.ndarray) -> None:
    try:
        text = await self._backend.transcribe(buffer, self._language)
        self._finish_transcription(text)
    except Exception as e:
        log.error("Transcription failed: %s", e)
        self._overlay.animate_exit(on_finished=self._overlay.hide)
        self._set_state(AppState.IDLE)
        self._send_notification("Voice Input Error", f"转写失败: {e}")

def _finish_transcription(self, text: str) -> None:
    if not text:
        self._overlay.animate_exit(on_finished=self._overlay.hide)
        self._set_state(AppState.IDLE)
        return

    if self._llm_enabled and self._llm is not None:
        self._set_state(AppState.REFINING)
        self._overlay.update_text("Refining...")
        asyncio.ensure_future(self._refine_and_inject(text))
    else:
        self._inject_and_finish(text)
```

- [ ] **Step 4: Update _on_language_changed to use new config path**

```python
@pyqtSlot(QAction)
def _on_language_changed(self, action: QAction) -> None:
    code = action.data()
    self._config["stt"]["local"]["language"] = code
    self._language = code
    self._whisper.language = code
    save_config(self._config)
    log.info("Language changed to: %s", code)
```

- [ ] **Step 5: Update tray set_state for TRANSCRIBING**

In `src/voice_input/tray.py`, update `set_state` to handle "Transcribing":

```python
def set_state(self, state: str) -> None:
    """Update tray state: 'Idle', 'Recording', 'Transcribing', 'Refining'."""
    self._state = state
    self._status_action.setText(f"Status: {state}")
    self.setToolTip(f"Voice Input — {state}")

    if state == "Recording":
        self._toggle_action.setText("Stop Recording")
        self.setIcon(self._icon_recording)
    elif state in ("Transcribing", "Refining"):
        self._toggle_action.setText("Stop Recording")
        self._toggle_action.setEnabled(False)
    else:
        self._toggle_action.setText("Start Recording")
        self._toggle_action.setEnabled(True)
        self.setIcon(self._icon_idle)
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass. Some existing tests in `test_whisper_worker.py` may need updating due to changed `WhisperWorker.__init__` signature — fix as needed.

- [ ] **Step 7: Commit**

```bash
git add src/voice_input/app.py src/voice_input/whisper_worker.py src/voice_input/tray.py
git commit -m "feat: integrate LocalBackend into AppController with non-streaming support"
```

---

### Task 9: Remove old local_whisper.py and update old tests

**Files:**
- Delete: `src/voice_input/backends/local_whisper.py`
- Modify: `tests/test_backend_local_whisper.py`
- Modify: `tests/test_whisper_worker.py`

- [ ] **Step 1: Update test_backend_local_whisper.py to test via LocalBackend**

Replace the file content to test `LocalBackend` with whisper engine (the old `LocalWhisperBackend` is now `LocalBackend` + `WhisperEngine`):

```python
# tests/test_backend_local_whisper.py
"""Tests for LocalBackend with whisper engine (replaces old LocalWhisperBackend tests)."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


def _whisper_config():
    return {
        "stt": {"backend": "local", "local": {
            "engine": "whisper", "model": "tiny",
            "language": "zh", "device": "cpu",
        }},
    }


def test_is_streaming_returns_true():
    from voice_input.backends.local import LocalBackend
    backend = LocalBackend(_whisper_config())
    assert backend.is_streaming() is True


@pytest.mark.asyncio
async def test_initialize_loads_model():
    from voice_input.backends.local import LocalBackend
    backend = LocalBackend(_whisper_config())
    mock_model = MagicMock()
    with patch("voice_input.backends.local.whisper_engine.WhisperModel", return_value=mock_model):
        await backend.initialize()
    assert backend._engine is not None
    assert backend._engine._model is mock_model


@pytest.mark.asyncio
async def test_transcribe_returns_text():
    from voice_input.backends.local import LocalBackend
    backend = LocalBackend(_whisper_config())
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], None)

    with patch("voice_input.backends.local.whisper_engine.WhisperModel", return_value=mock_model):
        await backend.initialize()
    audio = np.zeros(16000, dtype=np.int16)
    result = await backend.transcribe(audio, "zh")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_transcribe_short_audio_returns_empty():
    from voice_input.backends.local import LocalBackend
    backend = LocalBackend(_whisper_config())
    backend._engine = MagicMock()

    audio = np.zeros(100, dtype=np.int16)
    result = await backend.transcribe(audio, "zh")
    assert result == ""
```

- [ ] **Step 2: Update test_whisper_worker.py for new constructor signature**

Update the test file to use a mock backend instead of model_name/device params:

```python
# In tests/test_whisper_worker.py, update WhisperWorker construction:
# Old: WhisperWorker(whisper_queue=q, model_name="tiny", language="zh", device="cpu")
# New: WhisperWorker(whisper_queue=q, backend=mock_backend, language="zh")
```

Replace all `WhisperWorker(whisper_queue=..., model_name=..., language=..., device=...)` calls with `WhisperWorker(whisper_queue=..., backend=mock_backend, language=...)` where `mock_backend = MagicMock()`.

- [ ] **Step 3: Delete old local_whisper.py**

```bash
rm src/voice_input/backends/local_whisper.py
```

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove old LocalWhisperBackend, update tests for new architecture"
```

---

### Task 10: Update pyproject.toml dependencies

**Files:**
- Modify: `pyproject.toml:10-20`

- [ ] **Step 1: Update dependencies**

In `pyproject.toml`, remove `faster-whisper` from core dependencies and add optional dependency groups:

```toml
[project]
name = "voice-input"
version = "0.1.0"
description = "KDE Plasma 6 Wayland voice input via system tray"
requires-python = ">=3.11"
dependencies = [
    "PyQt6>=6.6",
    "sounddevice>=0.4",
    "dbus-next>=0.2",
    "httpx>=0.27",
    "qasync>=0.27",
    "keyring>=25.0",
    "numpy>=1.24",
    "tomli>=2.0; python_version < '3.12'",
]

[project.optional-dependencies]
whisper = ["faster-whisper>=1.0"]
sensevoice = ["funasr", "modelscope"]
local-all = ["faster-whisper>=1.0", "funasr", "modelscope"]
dev = ["pytest>=8.0", "pytest-qt>=4.3"]
```

- [ ] **Step 2: Verify project can still be parsed**

Run: `python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies'])"`
Expected: Prints the optional dependencies dict without errors

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "refactor: make whisper and sensevoice optional dependencies"
```

---

### Task 11: Add engine switch to tray menu

**Files:**
- Modify: `src/voice_input/tray.py`
- Modify: `src/voice_input/app.py` (connect new signal)

- [ ] **Step 1: Add engine submenu to TrayManager**

In `src/voice_input/tray.py`, add engine menu after the language menu section in `_build_menu()`:

```python
# After lang_menu section, before llm_menu:

# Engine submenu
engine_menu = QMenu("Local Engine", menu)
self._engine_group = QActionGroup(engine_menu)
self._engine_group.setExclusive(True)
self._engine_actions: dict[str, QAction] = {}
engines = {"whisper": "faster-whisper", "sensevoice": "SenseVoice Small"}
for key, label in engines.items():
    action = QAction(label, engine_menu)
    action.setCheckable(True)
    action.setData(key)
    self._engine_group.addAction(action)
    engine_menu.addAction(action)
    self._engine_actions[key] = action
menu.addMenu(engine_menu)
```

Add a property:
```python
@property
def engine_group(self) -> QActionGroup:
    return self._engine_group
```

Add a method to set current engine:
```python
def set_engine(self, engine: str) -> None:
    if engine in self._engine_actions:
        self._engine_actions[engine].setChecked(True)
```

- [ ] **Step 2: Connect engine switch in AppController**

In `src/voice_input/app.py`, in `_connect_signals()`:

```python
self._tray.engine_group.triggered.connect(self._on_engine_changed)
```

Add the handler:

```python
@pyqtSlot(QAction)
def _on_engine_changed(self, action: QAction) -> None:
    engine = action.data()
    if engine == self._config["stt"]["local"]["engine"]:
        return
    self._config["stt"]["local"]["engine"] = engine
    save_config(self._config)
    log.info("Engine changed to: %s", engine)
    # Reinitialize backend
    self._backend = create_backend(self._config)
    self._whisper.stop()
    self._whisper.wait()
    self._whisper = WhisperWorker(
        whisper_queue=self._whisper_queue,
        backend=self._backend,
        language=self._language,
    )
    self._whisper.transcription_updated.connect(self._on_transcription)
    self._whisper.error_occurred.connect(self._on_whisper_error)
    self._whisper.start()
```

In `start()`, set initial engine state:
```python
self._tray.set_engine(config.get("stt", {}).get("local", {}).get("engine", "whisper"))
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/voice_input/tray.py src/voice_input/app.py
git commit -m "feat: add engine switch menu in system tray"
```

---

### Task 12: Final integration test and cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Verify import chain works**

```bash
python -c "from voice_input.backends import create_backend; print('factory OK')"
python -c "from voice_input.backends.local import LocalBackend; print('LocalBackend OK')"
python -c "from voice_input.backends.local.whisper_engine import WhisperEngine; print('WhisperEngine OK')"
python -c "from voice_input.backends.local.sensevoice_engine import SenseVoiceEngine; print('SenseVoiceEngine OK')"
```

Expected: All print OK (SenseVoiceEngine will print OK even without funasr installed — the import guard handles it)

- [ ] **Step 3: Verify config round-trip**

```bash
python -c "
from voice_input.config import DEFAULT_CONFIG, _to_toml
import tomllib
toml_str = _to_toml(DEFAULT_CONFIG)
print(toml_str)
parsed = tomllib.loads(toml_str)
assert parsed['stt']['local']['engine'] == 'whisper'
print('Config round-trip OK')
"
```

- [ ] **Step 4: Final commit if any remaining changes**

```bash
git status
# If clean, skip. Otherwise:
git add -A
git commit -m "chore: final cleanup for SenseVoice engine support"
```
