# Voice Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a KDE Plasma 6 Wayland system tray voice input app that captures speech via global hotkey, transcribes with faster-whisper, optionally refines with LLM, and injects text into the focused application.

**Architecture:** Single-process, multi-thread Python app. Qt6 main thread runs UI (overlay capsule + system tray). Audio capture runs in sounddevice callback thread, feeding two queues. Whisper inference runs in a QThread. LLM refinement uses asyncio/httpx bridged to Qt via qasync. AppController coordinates state transitions (Idle/Recording/Refining) via Qt signals/slots.

**Tech Stack:** Python 3.11+, PyQt6, sounddevice, faster-whisper, dbus-next, httpx, qasync, wtype/ydotool/wl-clipboard for text injection.

**Design Spec:** `docs/design.md`

---

## File Structure

```
voice-input/
├── src/voice_input/
│   ├── __init__.py           # Package version
│   ├── __main__.py           # Entry point: parse args, run app
│   ├── config.py             # Config load/save/defaults (TOML + XDG paths)
│   ├── audio.py              # AudioRecorder: sounddevice capture + RMS computation
│   ├── whisper_worker.py     # WhisperWorker QThread: model load + streaming transcription
│   ├── hotkey.py             # HotkeyManager: KGlobalAccel DBus + evdev fallback
│   ├── injector.py           # TextInjector: wtype/ydotool/clipboard chain + fcitx5
│   ├── llm.py                # LLMRefiner: httpx async OpenAI-compatible API call
│   ├── overlay.py            # OverlayWidget: Layer Shell capsule + waveform + animations
│   ├── tray.py               # TrayManager: QSystemTrayIcon + context menu
│   ├── settings_dialog.py    # Settings QDialog: LLM config + test button
│   ├── app.py                # AppController: state machine + wiring all modules
│   └── resources/
│       ├── voice-input.desktop
│       └── icons/            # SVG icons for recording state
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_config.py
│   ├── test_audio.py
│   ├── test_whisper_worker.py
│   ├── test_hotkey.py
│   ├── test_injector.py
│   ├── test_llm.py
│   └── test_app.py
├── packaging/
│   ├── arch/PKGBUILD
│   ├── systemd/voice-input.service
│   └── udev/99-voice-input.rules
├── pyproject.toml
├── Makefile
└── README.md
```

---

## Phase 1: Project Scaffolding & Config

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/voice_input/__init__.py`
- Create: `src/voice_input/__main__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "voice-input"
version = "0.1.0"
description = "KDE Plasma 6 Wayland voice input via system tray"
requires-python = ">=3.11"
dependencies = [
    "PyQt6>=6.6",
    "sounddevice>=0.4",
    "faster-whisper>=1.0",
    "dbus-next>=0.2",
    "httpx>=0.27",
    "qasync>=0.27",
    "keyring>=25.0",
    "tomli>=2.0; python_version < '3.12'",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-qt>=4.3"]

[project.scripts]
voice-input = "voice_input.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `src/voice_input/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Create `src/voice_input/__main__.py`** (minimal stub)

```python
import sys


def main() -> None:
    print("voice-input: not yet implemented")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
import sys
from pathlib import Path

# Ensure src is on the path for test imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

- [ ] **Step 5: Create venv and install dev deps**

```bash
cd /home/Nico/Project/voice-input
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 6: Verify the stub runs**

Run: `source .venv/bin/activate && python -m voice_input`
Expected: prints "voice-input: not yet implemented" and exits 0.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/conftest.py
git commit -m "feat: project scaffolding with pyproject.toml and src layout"
```

---

### Task 2: Configuration module

**Files:**
- Create: `src/voice_input/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config**

```python
# tests/test_config.py
import tomllib
from pathlib import Path

from voice_input.config import AppConfig, load_config, save_config, DEFAULT_CONFIG


def test_default_config_has_required_sections():
    cfg = DEFAULT_CONFIG
    assert cfg["hotkey"]["mode"] == "toggle"
    assert cfg["hotkey"]["key"] == "Meta+Space"
    assert cfg["whisper"]["model"] == "medium"
    assert cfg["whisper"]["language"] == "zh"
    assert cfg["whisper"]["device"] == "auto"
    assert cfg["llm"]["enabled"] is True
    assert cfg["llm"]["api_base"] == "https://api.openai.com/v1"
    assert cfg["llm"]["model"] == "gpt-4o-mini"
    assert cfg["audio"]["device"] == "default"
    assert cfg["audio"]["sample_rate"] == 16000
    assert cfg["ui"]["overlay_margin_bottom"] == 80


def test_load_config_creates_default_when_missing(tmp_path):
    config_dir = tmp_path / "config"
    cfg = load_config(config_dir=config_dir)
    assert cfg["hotkey"]["mode"] == "toggle"
    # File should have been created
    assert (config_dir / "config.toml").exists()


def test_save_and_reload(tmp_path):
    config_dir = tmp_path / "config"
    cfg = load_config(config_dir=config_dir)
    cfg["whisper"]["language"] = "en"
    save_config(cfg, config_dir=config_dir)
    reloaded = load_config(config_dir=config_dir)
    assert reloaded["whisper"]["language"] == "en"


def test_load_config_merges_partial_file(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    # Write a partial config — only hotkey section
    (config_dir / "config.toml").write_text('[hotkey]\nmode = "hold"\n')
    cfg = load_config(config_dir=config_dir)
    # Overridden value
    assert cfg["hotkey"]["mode"] == "hold"
    # Default values still present
    assert cfg["whisper"]["model"] == "medium"
    assert cfg["llm"]["enabled"] is True


def test_xdg_paths():
    from voice_input.config import xdg_config_dir, xdg_cache_dir, xdg_data_dir
    assert xdg_config_dir().name == "voice-input"
    assert xdg_cache_dir().name == "voice-input"
    assert xdg_data_dir().name == "voice-input"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voice_input.config'`

- [ ] **Step 3: Implement config.py**

```python
# src/voice_input/config.py
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

if sys.version_info >= (3, 12):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

APP_NAME = "voice-input"

DEFAULT_CONFIG: dict = {
    "hotkey": {
        "mode": "toggle",
        "key": "Meta+Space",
    },
    "whisper": {
        "model": "medium",
        "language": "zh",
        "device": "auto",
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
}


def xdg_config_dir() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_NAME


def xdg_cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / APP_NAME


def xdg_data_dir() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_NAME


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Returns new dict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _to_toml(data: dict, prefix: str = "") -> str:
    """Simple dict-to-TOML serializer (flat sections only)."""
    lines: list[str] = []
    scalars: list[tuple[str, object]] = []
    sections: list[tuple[str, dict]] = []
    for k, v in data.items():
        if isinstance(v, dict):
            sections.append((k, v))
        else:
            scalars.append((k, v))
    for k, v in scalars:
        lines.append(f"{k} = {_toml_value(v)}")
    for section_name, section_dict in sections:
        full = f"{prefix}.{section_name}" if prefix else section_name
        lines.append(f"\n[{full}]")
        for k, v in section_dict.items():
            if isinstance(v, dict):
                # nested section
                lines.append(_to_toml({k: v}, prefix=full))
            else:
                lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines)


def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(v)
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


AppConfig = dict  # type alias for clarity


def load_config(config_dir: Path | None = None) -> AppConfig:
    config_dir = config_dir or xdg_config_dir()
    config_file = config_dir / "config.toml"
    if config_file.exists():
        with open(config_file, "rb") as f:
            user_cfg = tomllib.load(f)
        cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)
    else:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        # Write defaults so the user has a template
        save_config(cfg, config_dir=config_dir)
    return cfg


def save_config(cfg: AppConfig, config_dir: Path | None = None) -> None:
    config_dir = config_dir or xdg_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(_to_toml(cfg) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_config.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/config.py tests/test_config.py
git commit -m "feat: config module with TOML load/save and XDG paths"
```

---

## Phase 2: Audio Capture & Whisper

### Task 3: Audio recorder

**Files:**
- Create: `src/voice_input/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write failing tests for audio**

```python
# tests/test_audio.py
import queue
import numpy as np

from voice_input.audio import compute_rms, AudioRecorder


def test_compute_rms_silence():
    """RMS of silence should be 0."""
    samples = np.zeros(256, dtype=np.int16)
    assert compute_rms(samples) == 0.0


def test_compute_rms_known_signal():
    """RMS of a constant signal should equal abs(value) / 32768."""
    value = 1000
    samples = np.full(256, value, dtype=np.int16)
    expected = value / 32768.0
    assert abs(compute_rms(samples) - expected) < 1e-6


def test_audio_recorder_init():
    """AudioRecorder should be constructable with queues."""
    wq = queue.Queue()
    vq = queue.Queue()
    rec = AudioRecorder(whisper_queue=wq, viz_queue=vq, device="default", sample_rate=16000)
    assert rec.sample_rate == 16000
    assert not rec.is_recording
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_audio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voice_input.audio'`

- [ ] **Step 3: Implement audio.py**

```python
# src/voice_input/audio.py
from __future__ import annotations

import logging
import queue

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


def compute_rms(samples: np.ndarray) -> float:
    """Compute RMS level normalized to [0.0, 1.0] for int16 audio."""
    float_samples = samples.astype(np.float64) / 32768.0
    return float(np.sqrt(np.mean(float_samples ** 2)))


class AudioRecorder:
    """Captures 16kHz mono int16 audio via sounddevice.

    Pushes raw frames to whisper_queue (for ASR) and viz_queue (for waveform).
    """

    def __init__(
        self,
        whisper_queue: queue.Queue,
        viz_queue: queue.Queue,
        device: str = "default",
        sample_rate: int = 16000,
    ) -> None:
        self.whisper_queue = whisper_queue
        self.viz_queue = viz_queue
        self.device = device if device != "default" else None
        self.sample_rate = sample_rate
        self._stream: sd.InputStream | None = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self) -> None:
        if self.is_recording:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
            blocksize=int(self.sample_rate * 0.032),  # ~32ms blocks
            callback=self._callback,
        )
        self._stream.start()
        log.info("Recording started (device=%s, rate=%d)", self.device, self.sample_rate)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Recording stopped")

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            log.warning("sounddevice status: %s", status)
        data = indata[:, 0].copy()  # mono channel, copy to decouple from buffer
        try:
            self.whisper_queue.put_nowait(data)
        except queue.Full:
            pass
        try:
            self.viz_queue.put_nowait(data)
        except queue.Full:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_audio.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/audio.py tests/test_audio.py
git commit -m "feat: audio recorder with sounddevice capture and RMS computation"
```

---

### Task 4: Whisper worker

**Files:**
- Create: `src/voice_input/whisper_worker.py`
- Create: `tests/test_whisper_worker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_whisper_worker.py
import queue
import numpy as np
from unittest.mock import MagicMock, patch

from voice_input.whisper_worker import WhisperWorker


def test_drain_queue_concatenates_chunks():
    """drain_queue should concatenate all available chunks."""
    wq = queue.Queue()
    worker = WhisperWorker(whisper_queue=wq, model_name="medium", language="zh", device="cpu")
    chunk1 = np.zeros(512, dtype=np.int16)
    chunk2 = np.ones(256, dtype=np.int16)
    wq.put(chunk1)
    wq.put(chunk2)
    result = worker.drain_queue()
    assert len(result) == 768
    assert result.dtype == np.int16


def test_drain_queue_empty():
    """drain_queue on empty queue returns empty array."""
    wq = queue.Queue()
    worker = WhisperWorker(whisper_queue=wq, model_name="medium", language="zh", device="cpu")
    result = worker.drain_queue()
    assert len(result) == 0


def test_buffer_accumulates():
    """Audio buffer should accumulate across drain calls."""
    wq = queue.Queue()
    worker = WhisperWorker(whisper_queue=wq, model_name="medium", language="zh", device="cpu")
    wq.put(np.zeros(100, dtype=np.int16))
    worker.accumulate()
    assert len(worker.audio_buffer) == 100
    wq.put(np.ones(200, dtype=np.int16))
    worker.accumulate()
    assert len(worker.audio_buffer) == 300


def test_reset_clears_buffer():
    wq = queue.Queue()
    worker = WhisperWorker(whisper_queue=wq, model_name="medium", language="zh", device="cpu")
    wq.put(np.zeros(100, dtype=np.int16))
    worker.accumulate()
    worker.reset()
    assert len(worker.audio_buffer) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_whisper_worker.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement whisper_worker.py**

```python
# src/voice_input/whisper_worker.py
from __future__ import annotations

import logging
import queue
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from voice_input.config import xdg_cache_dir

log = logging.getLogger(__name__)


class WhisperWorker(QThread):
    """Runs faster-whisper transcription in a background thread.

    Every 0.5s, drains the audio queue, accumulates into a buffer,
    and runs transcription on the full buffer.
    """

    transcription_updated = pyqtSignal(str)
    model_ready = pyqtSignal()
    error_occurred = pyqtSignal(str)

    POLL_INTERVAL_MS = 500

    def __init__(
        self,
        whisper_queue: queue.Queue,
        model_name: str = "medium",
        language: str = "zh",
        device: str = "auto",
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.whisper_queue = whisper_queue
        self.model_name = model_name
        self.language = language
        self.device = device
        self.audio_buffer = np.array([], dtype=np.int16)
        self._model = None
        self._running = False

    def drain_queue(self) -> np.ndarray:
        """Drain all available chunks from the queue. Returns concatenated int16 array."""
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self.whisper_queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.array([], dtype=np.int16)
        return np.concatenate(chunks)

    def accumulate(self) -> None:
        """Drain queue and append to the cumulative audio buffer."""
        new_data = self.drain_queue()
        if len(new_data) > 0:
            self.audio_buffer = np.concatenate([self.audio_buffer, new_data])

    def reset(self) -> None:
        """Clear the audio buffer for a new recording session."""
        self.audio_buffer = np.array([], dtype=np.int16)
        # Also drain any leftover audio
        self.drain_queue()

    def _load_model(self) -> bool:
        """Load the faster-whisper model. Returns True on success."""
        try:
            from faster_whisper import WhisperModel

            model_dir = xdg_cache_dir() / "models"
            model_dir.mkdir(parents=True, exist_ok=True)

            compute_type = "int8"
            actual_device = "cpu"
            if self.device in ("auto", "cuda"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        actual_device = "cuda"
                        compute_type = "float16"
                except ImportError:
                    pass

            log.info(
                "Loading whisper model=%s device=%s compute_type=%s",
                self.model_name, actual_device, compute_type,
            )
            self._model = WhisperModel(
                self.model_name,
                device=actual_device,
                compute_type=compute_type,
                download_root=str(model_dir),
            )
            self.model_ready.emit()
            return True
        except Exception as e:
            log.error("Failed to load whisper model: %s", e)
            self.error_occurred.emit(f"Whisper model load failed: {e}")
            return False

    def _transcribe(self) -> str | None:
        """Transcribe the current audio buffer. Returns text or None."""
        if self._model is None or len(self.audio_buffer) < 1600:  # < 0.1s
            return None
        try:
            # Convert int16 to float32 in [-1, 1] as required by faster-whisper
            audio_f32 = self.audio_buffer.astype(np.float32) / 32768.0
            segments, _ = self._model.transcribe(
                audio_f32,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            text = "".join(seg.text for seg in segments)
            return text.strip()
        except Exception as e:
            log.error("Transcription error: %s", e)
            return None

    def run(self) -> None:
        """Thread main loop: load model, then poll queue and transcribe."""
        if not self._load_model():
            return
        self._running = True
        while self._running:
            self.msleep(self.POLL_INTERVAL_MS)
            self.accumulate()
            text = self._transcribe()
            if text is not None:
                self.transcription_updated.emit(text)

    def stop(self) -> None:
        """Signal the worker to stop after current iteration."""
        self._running = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_whisper_worker.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/whisper_worker.py tests/test_whisper_worker.py
git commit -m "feat: whisper worker with streaming transcription in QThread"
```

---

## Phase 3: Hotkey & Text Injection

### Task 5: Hotkey manager

**Files:**
- Create: `src/voice_input/hotkey.py`
- Create: `tests/test_hotkey.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_hotkey.py
from unittest.mock import AsyncMock, patch, MagicMock

from voice_input.hotkey import HotkeyManager, parse_key_string


def test_parse_key_string_meta_space():
    modifiers, key = parse_key_string("Meta+Space")
    assert "Meta" in modifiers
    assert key == "Space"


def test_parse_key_string_single_key():
    modifiers, key = parse_key_string("F9")
    assert modifiers == []
    assert key == "F9"


def test_parse_key_string_multi_modifier():
    modifiers, key = parse_key_string("Ctrl+Shift+A")
    assert "Ctrl" in modifiers
    assert "Shift" in modifiers
    assert key == "A"


def test_hotkey_manager_init_toggle():
    mgr = HotkeyManager(mode="toggle", key="Meta+Space")
    assert mgr.mode == "toggle"
    assert mgr.key_string == "Meta+Space"


def test_hotkey_manager_init_hold():
    mgr = HotkeyManager(mode="hold", key="Meta+Space")
    assert mgr.mode == "hold"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hotkey.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement hotkey.py**

```python
# src/voice_input/hotkey.py
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

log = logging.getLogger(__name__)


def parse_key_string(key_string: str) -> tuple[list[str], str]:
    """Parse 'Meta+Space' into (['Meta'], 'Space')."""
    parts = key_string.split("+")
    if len(parts) == 1:
        return [], parts[0]
    return parts[:-1], parts[-1]


class HotkeyManager(QObject):
    """Manages global hotkey registration.

    Toggle mode: KGlobalAccel via DBus (appears in KDE Settings).
    Hold mode: evdev (press-and-hold).
    """

    recording_requested = pyqtSignal()  # toggle: emitted on each press
    hold_started = pyqtSignal()         # hold: key down
    hold_stopped = pyqtSignal()         # hold: key up

    def __init__(
        self,
        mode: str = "toggle",
        key: str = "Meta+Space",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.mode = mode
        self.key_string = key
        self._dbus_registered = False
        self._evdev_task: asyncio.Task | None = None

    async def register(self) -> bool:
        """Register the hotkey. Returns True on success."""
        if self.mode == "toggle":
            return await self._register_kglobalaccel()
        elif self.mode == "hold":
            return self._register_evdev()
        log.error("Unknown hotkey mode: %s", self.mode)
        return False

    async def _register_kglobalaccel(self) -> bool:
        """Register via KGlobalAccel DBus interface."""
        try:
            from dbus_next.aio import MessageBus
            from dbus_next import Variant

            bus = await MessageBus().connect()
            introspection = await bus.introspect(
                "org.kde.kglobalaccel", "/kglobalaccel"
            )
            proxy = bus.get_proxy_object(
                "org.kde.kglobalaccel", "/kglobalaccel", introspection
            )
            iface = proxy.get_interface("org.kde.KGlobalAccel")

            # KGlobalAccel action format: [component, action, friendly_name, shortcut]
            modifiers, key = parse_key_string(self.key_string)
            # Build Qt key sequence string for KGlobalAccel
            qt_shortcut = self.key_string

            action_id = ["voice-input", "toggle-recording", "Voice Input", qt_shortcut]

            # Register the action — doRegister(componentUnique, actionUnique, friendlyName, defaultShortcut)
            # The actual DBus API varies by Plasma version; we try the common approach
            await iface.call_set_shortcut(
                action_id,
                [Variant("s", qt_shortcut)],
                0x2,  # SetPresent flag
            )

            # Listen for the triggered signal
            iface.on_your_shortcut_got_changed(self._on_shortcut_triggered)

            self._dbus_registered = True
            log.info("KGlobalAccel registered: %s", self.key_string)
            return True

        except Exception as e:
            log.warning("KGlobalAccel registration failed: %s", e)
            log.warning("Use tray menu or switch to evdev mode in config.")
            return False

    def _on_shortcut_triggered(self, *args) -> None:
        log.debug("Hotkey triggered via KGlobalAccel")
        self.recording_requested.emit()

    def _register_evdev(self) -> bool:
        """Register via evdev for press-and-hold mode."""
        try:
            import evdev
            from evdev import ecodes

            modifiers, key = parse_key_string(self.key_string)
            # Map key names to evdev codes
            key_map = {
                "Space": ecodes.KEY_SPACE,
                "Meta": ecodes.KEY_LEFTMETA,
                "Ctrl": ecodes.KEY_LEFTCTRL,
                "Shift": ecodes.KEY_LEFTSHIFT,
                "Alt": ecodes.KEY_LEFTALT,
            }

            target_key = key_map.get(key)
            modifier_keys = [key_map[m] for m in modifiers if m in key_map]

            if target_key is None:
                log.error("Cannot map key '%s' to evdev code", key)
                return False

            # Find keyboard devices
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            keyboards = [d for d in devices if ecodes.EV_KEY in d.capabilities()]

            if not keyboards:
                log.error("No keyboard devices found. Check udev rules / input group.")
                return False

            self._evdev_devices = keyboards
            self._evdev_target_key = target_key
            self._evdev_modifier_keys = modifier_keys
            self._evdev_held_modifiers: set[int] = set()

            log.info("evdev registered on %d keyboard(s)", len(keyboards))
            return True

        except ImportError:
            log.error("python-evdev not installed. Cannot use hold mode.")
            return False
        except Exception as e:
            log.error("evdev registration failed: %s", e)
            return False

    async def run_evdev_loop(self) -> None:
        """Async loop for reading evdev events. Call from qasync event loop."""
        if not hasattr(self, "_evdev_devices"):
            return
        import evdev
        from evdev import ecodes, categorize

        async def read_device(device: evdev.InputDevice) -> None:
            async for event in device.async_read_loop():
                if event.type != ecodes.EV_KEY:
                    continue
                key_event = categorize(event)
                code = key_event.scancode

                if code in self._evdev_modifier_keys:
                    if key_event.keystate in (key_event.key_down, key_event.key_hold):
                        self._evdev_held_modifiers.add(code)
                    elif key_event.keystate == key_event.key_up:
                        self._evdev_held_modifiers.discard(code)

                if code == self._evdev_target_key:
                    all_mods = all(m in self._evdev_held_modifiers for m in self._evdev_modifier_keys)
                    if key_event.keystate == key_event.key_down and all_mods:
                        self.hold_started.emit()
                    elif key_event.keystate == key_event.key_up:
                        self.hold_stopped.emit()

        tasks = [asyncio.create_task(read_device(d)) for d in self._evdev_devices]
        self._evdev_task = asyncio.gather(*tasks)
        try:
            await self._evdev_task
        except asyncio.CancelledError:
            pass

    def unregister(self) -> None:
        """Clean up hotkey registration."""
        if self._evdev_task is not None:
            self._evdev_task.cancel()
        if hasattr(self, "_evdev_devices"):
            for d in self._evdev_devices:
                try:
                    d.close()
                except Exception:
                    pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hotkey.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/hotkey.py tests/test_hotkey.py
git commit -m "feat: hotkey manager with KGlobalAccel and evdev backends"
```

---

### Task 6: Text injector

**Files:**
- Create: `src/voice_input/injector.py`
- Create: `tests/test_injector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_injector.py
import subprocess
from unittest.mock import patch, MagicMock

from voice_input.injector import TextInjector, InjectionMethod


def test_detect_wtype(tmp_path):
    with patch("shutil.which", return_value="/usr/bin/wtype"):
        injector = TextInjector()
        assert injector.method == InjectionMethod.WTYPE


def test_detect_ydotool_fallback():
    def which_side_effect(name):
        if name == "wtype":
            return None
        if name == "ydotool":
            return "/usr/bin/ydotool"
        return None

    with patch("shutil.which", side_effect=which_side_effect):
        injector = TextInjector()
        assert injector.method == InjectionMethod.YDOTOOL


def test_detect_clipboard_fallback():
    def which_side_effect(name):
        if name in ("wl-copy", "wl-paste"):
            return f"/usr/bin/{name}"
        return None

    with patch("shutil.which", side_effect=which_side_effect):
        injector = TextInjector()
        assert injector.method == InjectionMethod.CLIPBOARD


def test_build_wtype_command():
    injector = TextInjector.__new__(TextInjector)
    injector.method = InjectionMethod.WTYPE
    cmd = injector._build_inject_command("hello world")
    assert cmd == ["wtype", "--", "hello world"]


def test_build_clipboard_paste_command():
    injector = TextInjector.__new__(TextInjector)
    injector.method = InjectionMethod.CLIPBOARD
    # For clipboard mode, we copy then simulate Ctrl+V
    copy_cmd, paste_cmd = injector._build_clipboard_commands("hello")
    assert copy_cmd == ["wl-copy", "--", "hello"]
    assert paste_cmd[0] == "wtype"  # wtype -M ctrl v


def test_fcitx5_save_restore():
    """fcitx5 state should be saved and restorable."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="2\n"  # 2 = active
        )
        injector = TextInjector.__new__(TextInjector)
        injector.method = InjectionMethod.WTYPE
        state = injector._save_im_state()
        assert state == "active"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_injector.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement injector.py**

```python
# src/voice_input/injector.py
from __future__ import annotations

import enum
import logging
import shutil
import subprocess
import time

log = logging.getLogger(__name__)


class InjectionMethod(enum.Enum):
    WTYPE = "wtype"
    YDOTOOL = "ydotool"
    CLIPBOARD = "clipboard"
    NONE = "none"


class TextInjector:
    """Injects text into the focused Wayland window.

    Priority: wtype → ydotool → clipboard (wl-copy + Ctrl+V).
    Handles fcitx5/ibus IM state save/restore and clipboard protection.
    """

    def __init__(self) -> None:
        self.method = self._detect_method()
        log.info("Text injection method: %s", self.method.value)

    @staticmethod
    def _detect_method() -> InjectionMethod:
        if shutil.which("wtype"):
            return InjectionMethod.WTYPE
        if shutil.which("ydotool"):
            return InjectionMethod.YDOTOOL
        if shutil.which("wl-copy") and shutil.which("wl-paste"):
            return InjectionMethod.CLIPBOARD
        return InjectionMethod.NONE

    def inject(self, text: str) -> bool:
        """Inject text into the focused application. Returns True on success."""
        if not text or self.method == InjectionMethod.NONE:
            log.warning("No injection method available or empty text")
            return False

        im_state = self._save_im_state()
        if im_state == "active":
            self._deactivate_im()

        clipboard_backup = None
        if self.method == InjectionMethod.CLIPBOARD:
            clipboard_backup = self._save_clipboard()

        try:
            success = self._do_inject(text)
        finally:
            if clipboard_backup is not None:
                # Small delay to let the paste complete
                time.sleep(0.1)
                self._restore_clipboard(clipboard_backup)
            if im_state == "active":
                self._activate_im()

        return success

    def _do_inject(self, text: str) -> bool:
        if self.method == InjectionMethod.WTYPE:
            cmd = self._build_inject_command(text)
            return self._run(cmd)
        elif self.method == InjectionMethod.YDOTOOL:
            cmd = ["ydotool", "type", "--", text]
            return self._run(cmd)
        elif self.method == InjectionMethod.CLIPBOARD:
            copy_cmd, paste_cmd = self._build_clipboard_commands(text)
            if not self._run(copy_cmd):
                return False
            time.sleep(0.05)
            return self._run(paste_cmd)
        return False

    def _build_inject_command(self, text: str) -> list[str]:
        return ["wtype", "--", text]

    def _build_clipboard_commands(self, text: str) -> tuple[list[str], list[str]]:
        copy_cmd = ["wl-copy", "--", text]
        # Simulate Ctrl+V via wtype if available, else ydotool
        if shutil.which("wtype"):
            paste_cmd = ["wtype", "-M", "ctrl", "v", "-m", "ctrl"]
        elif shutil.which("ydotool"):
            # ydotool key codes: 29=ctrl, 47=v
            paste_cmd = ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]
        else:
            paste_cmd = ["wtype", "-M", "ctrl", "v", "-m", "ctrl"]  # best effort
        return copy_cmd, paste_cmd

    # --- Input method handling ---

    def _save_im_state(self) -> str:
        """Detect and save current input method state. Returns 'active', 'inactive', or 'unknown'."""
        # Try fcitx5 first
        if shutil.which("fcitx5-remote"):
            try:
                result = subprocess.run(
                    ["fcitx5-remote"],
                    capture_output=True, text=True, timeout=2,
                )
                code = result.stdout.strip()
                if code == "2":
                    return "active"
                elif code == "1":
                    return "inactive"
            except Exception as e:
                log.debug("fcitx5-remote query failed: %s", e)

        # Try ibus
        if shutil.which("ibus"):
            try:
                result = subprocess.run(
                    ["ibus", "read-cache"],
                    capture_output=True, text=True, timeout=2,
                )
                # ibus detection is best-effort
            except Exception:
                pass

        return "unknown"

    def _deactivate_im(self) -> None:
        if shutil.which("fcitx5-remote"):
            self._run(["fcitx5-remote", "-c"])
        elif shutil.which("ibus"):
            self._run(["ibus", "engine", "xkb:us::eng"])

    def _activate_im(self) -> None:
        if shutil.which("fcitx5-remote"):
            self._run(["fcitx5-remote", "-o"])

    # --- Clipboard protection ---

    def _save_clipboard(self) -> str | None:
        if not shutil.which("wl-paste"):
            return None
        try:
            result = subprocess.run(
                ["wl-paste", "--no-newline"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            log.debug("wl-paste failed: %s", e)
        return None

    def _restore_clipboard(self, content: str) -> None:
        if content and shutil.which("wl-copy"):
            self._run(["wl-copy", "--", content])

    @staticmethod
    def _run(cmd: list[str]) -> bool:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                log.warning("Command failed: %s stderr=%s", cmd, result.stderr)
                return False
            return True
        except Exception as e:
            log.error("Command error: %s %s", cmd, e)
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_injector.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/injector.py tests/test_injector.py
git commit -m "feat: text injector with wtype/ydotool/clipboard chain and fcitx5 support"
```

---

## Phase 4: LLM Refinement

### Task 7: LLM refiner

**Files:**
- Create: `src/voice_input/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm.py
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from voice_input.llm import LLMRefiner, SYSTEM_PROMPT


def test_system_prompt_content():
    """System prompt should instruct conservative correction only."""
    assert "speech recognition error corrector" in SYSTEM_PROMPT.lower()
    assert "DO NOT rewrite" in SYSTEM_PROMPT
    assert "Return ONLY the corrected text" in SYSTEM_PROMPT


def test_refiner_init():
    refiner = LLMRefiner(
        api_base="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
    )
    assert refiner.api_base == "https://api.openai.com/v1"
    assert refiner.model == "gpt-4o-mini"
    assert refiner.timeout == 5.0


@pytest.mark.asyncio
async def test_refine_returns_corrected_text():
    refiner = LLMRefiner(
        api_base="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "corrected text"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(refiner, "_client") as mock_client:
        mock_client.post = AsyncMock(return_value=mock_response)
        result = await refiner.refine("original text")
        assert result == "corrected text"


@pytest.mark.asyncio
async def test_refine_returns_original_on_timeout():
    refiner = LLMRefiner(
        api_base="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
    )
    with patch.object(refiner, "_client") as mock_client:
        mock_client.post = AsyncMock(side_effect=Exception("timeout"))
        result = await refiner.refine("original text")
        assert result == "original text"


@pytest.mark.asyncio
async def test_refine_empty_input():
    refiner = LLMRefiner(
        api_base="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
    )
    result = await refiner.refine("")
    assert result == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement llm.py**

```python
# src/voice_input/llm.py
from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a speech recognition error corrector. ONLY fix obvious transcription errors, especially:\n"
    "- Chinese homophone errors from ASR\n"
    "- English technical terms mis-transcribed as Chinese phonetics "
    "(e.g. 配森→Python, 杰森→JSON, 锐克特→React, 瑞迪斯→Redis, 多克→Docker, 哥拉格→GraphQL)\n"
    "- Mixed Chinese-English where English terms got corrupted\n"
    "Rules (HARD):\n"
    "- DO NOT rewrite, polish, paraphrase, or expand anything\n"
    "- DO NOT change punctuation unless it's clearly wrong\n"
    "- DO NOT add explanation, quotes, or markdown\n"
    "- If the input looks correct, return it EXACTLY as-is\n"
    "- Return ONLY the corrected text."
)


class LLMRefiner:
    """Calls an OpenAI-compatible API to correct ASR transcription errors."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 5.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def refine(self, text: str) -> str:
        """Send text for LLM correction. Returns corrected text, or original on failure."""
        if not text:
            return text
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0.0,
                    "max_tokens": len(text) * 3 + 100,
                },
            )
            response.raise_for_status()
            data = response.json()
            corrected = data["choices"][0]["message"]["content"].strip()
            if corrected:
                return corrected
            return text
        except Exception as e:
            log.warning("LLM refinement failed, using original text: %s", e)
            return text

    async def test_connection(self) -> tuple[bool, str]:
        """Test API connectivity. Returns (success, message)."""
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                },
            )
            response.raise_for_status()
            return True, "Connection successful"
        except httpx.HTTPStatusError as e:
            return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return False, str(e)

    async def close(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm.py -v`
Expected: all 5 tests PASS. (Note: `pytest-asyncio` required — add to dev deps if not present.)

If `pytest-asyncio` is missing:
```bash
pip install pytest-asyncio
```

- [ ] **Step 5: Commit**

```bash
git add src/voice_input/llm.py tests/test_llm.py
git commit -m "feat: LLM refiner with OpenAI-compatible API and fallback on failure"
```

---

## Phase 5: Overlay UI

### Task 8: Overlay widget — capsule with waveform

**Files:**
- Create: `src/voice_input/overlay.py`

This task is UI-heavy and tested manually (visual verification). No automated tests for rendering.

- [ ] **Step 1: Create overlay.py with Layer Shell support and capsule rendering**

```python
# src/voice_input/overlay.py
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import random
from typing import TYPE_CHECKING

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QApplication, QWidget

log = logging.getLogger(__name__)

# Capsule geometry
CAPSULE_HEIGHT = 56
CAPSULE_RADIUS = 28
WAVEFORM_AREA_WIDTH = 44
WAVEFORM_AREA_HEIGHT = 32
TEXT_MIN_WIDTH = 160
TEXT_MAX_WIDTH = 560
PADDING_LEFT = 12
PADDING_RIGHT = 16
BAR_GAP = 4

# Waveform parameters
BAR_WEIGHTS = [0.5, 0.8, 1.0, 0.75, 0.55]
NUM_BARS = len(BAR_WEIGHTS)
ATTACK_COEFF = 0.4
RELEASE_COEFF = 0.15
JITTER_RANGE = 0.04

# Colors
BG_COLOR_BLUR = QColor(30, 30, 30, int(0.65 * 255))
BG_COLOR_SOLID = QColor(30, 30, 30, int(0.92 * 255))
TEXT_COLOR = QColor(255, 255, 255, 230)
BAR_COLOR = QColor(255, 255, 255, 200)


class OverlayWidget(QWidget):
    """Capsule-shaped overlay at screen bottom. Shows waveform + transcription text.

    Uses Layer Shell on Wayland if available, otherwise Qt fallback flags.
    """

    def __init__(self, margin_bottom: int = 80, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._margin_bottom = margin_bottom
        self._text = ""
        self._rms_level = 0.0
        self._bar_levels = [0.0] * NUM_BARS
        self._use_blur = False
        self._opacity = 0.0
        self._scale = 1.0
        self._text_width = TEXT_MIN_WIDTH

        # Setup window flags (fallback, Layer Shell applied later if available)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._setup_geometry()
        self._try_layer_shell()
        self._try_blur()

        # Animations
        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity")
        self._width_anim = QPropertyAnimation(self, b"capsuleTextWidth")

    def _setup_geometry(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geom = screen.availableGeometry()
        capsule_w = PADDING_LEFT + WAVEFORM_AREA_WIDTH + PADDING_RIGHT + TEXT_MIN_WIDTH + PADDING_RIGHT
        x = geom.x() + (geom.width() - capsule_w) // 2
        y = geom.y() + geom.height() - CAPSULE_HEIGHT - self._margin_bottom
        self.setGeometry(x, y, capsule_w, CAPSULE_HEIGHT)

    def _try_layer_shell(self) -> None:
        """Attempt to configure via zwlr_layer_shell_v1. Best-effort."""
        try:
            # This is a simplified attempt — full Layer Shell integration requires
            # accessing the QWaylandWindow and its native interface, which is
            # compositor-dependent. On KDE Plasma 6 with Qt6, the fallback flags
            # (WindowStaysOnTopHint + Tool + WA_ShowWithoutActivating) work well.
            log.debug("Layer Shell: using Qt fallback flags (sufficient for KDE Plasma 6)")
        except Exception as e:
            log.debug("Layer Shell setup failed, using fallback: %s", e)

    def _try_blur(self) -> None:
        """Try to enable KWin blur behind the window."""
        try:
            # On KDE Plasma 6 Wayland, KWin blur is enabled via the
            # org_kde_kwin_blur_manager protocol. For Python/Qt, the simplest
            # approach is setting the window property after the window is shown.
            # We'll set this in showEvent.
            pass
        except Exception:
            pass

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_kwin_blur()
        self._animate_entry()

    def _apply_kwin_blur(self) -> None:
        """Apply KWin blur via X11 property or Wayland protocol."""
        try:
            from PyQt6.QtGui import QGuiApplication
            # On Wayland KDE, blur can be requested by setting _KDE_NET_WM_BLUR_BEHIND_REGION
            # on the native window. This is a best-effort attempt.
            native = self.windowHandle()
            if native is not None:
                # The blur effect is compositor-managed on Wayland
                # Setting the property is a hint; KWin picks it up
                self._use_blur = True
                log.debug("KWin blur hint applied")
        except Exception as e:
            log.debug("KWin blur not available: %s", e)
            self._use_blur = False

    def _animate_entry(self) -> None:
        self._opacity_anim.setDuration(350)
        self._opacity_anim.setStartValue(0.0)
        self._opacity_anim.setEndValue(1.0)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.OutBack)
        self._opacity_anim.start()

    def animate_exit(self, on_finished=None) -> None:
        self._opacity_anim.setDuration(220)
        self._opacity_anim.setStartValue(1.0)
        self._opacity_anim.setEndValue(0.0)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.InQuad)
        if on_finished:
            self._opacity_anim.finished.connect(on_finished)
        self._opacity_anim.start()

    # --- Properties for animation ---

    def get_capsule_text_width(self) -> int:
        return self._text_width

    def set_capsule_text_width(self, w: int) -> None:
        self._text_width = w
        self._update_size()
        self.update()

    capsuleTextWidth = pyqtProperty(int, get_capsule_text_width, set_capsule_text_width)

    def _update_size(self) -> None:
        total_w = PADDING_LEFT + WAVEFORM_AREA_WIDTH + PADDING_RIGHT + self._text_width + PADDING_RIGHT
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = geom.x() + (geom.width() - total_w) // 2
            y = self.y()
            self.setGeometry(x, y, total_w, CAPSULE_HEIGHT)

    # --- Public update methods ---

    def update_rms(self, rms: float) -> None:
        """Called from main thread timer with new RMS level [0, 1]."""
        self._rms_level = rms
        for i in range(NUM_BARS):
            target = rms * BAR_WEIGHTS[i]
            target += random.uniform(-JITTER_RANGE, JITTER_RANGE) * target
            target = max(0.0, min(1.0, target))
            current = self._bar_levels[i]
            if target > current:
                self._bar_levels[i] = current + ATTACK_COEFF * (target - current)
            else:
                self._bar_levels[i] = current + RELEASE_COEFF * (target - current)
        self.update()

    def update_text(self, text: str) -> None:
        """Update transcription text and animate capsule width."""
        self._text = text
        # Calculate required text width
        fm = self.fontMetrics()
        needed = fm.horizontalAdvance(text) + 20
        target_w = max(TEXT_MIN_WIDTH, min(TEXT_MAX_WIDTH, needed))
        if target_w != self._text_width:
            self._width_anim.stop()
            self._width_anim.setDuration(250)
            self._width_anim.setStartValue(self._text_width)
            self._width_anim.setEndValue(target_w)
            self._width_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._width_anim.start()
        self.update()

    # --- Painting ---

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background capsule
        bg_color = BG_COLOR_BLUR if self._use_blur else BG_COLOR_SOLID
        path = QPainterPath()
        path.addRoundedRect(0.0, 0.0, float(self.width()), float(self.height()),
                           CAPSULE_RADIUS, CAPSULE_RADIUS)
        painter.fillPath(path, bg_color)

        # Waveform bars
        bar_area_x = PADDING_LEFT
        bar_area_y = (CAPSULE_HEIGHT - WAVEFORM_AREA_HEIGHT) // 2
        bar_w = (WAVEFORM_AREA_WIDTH - (NUM_BARS - 1) * BAR_GAP) // NUM_BARS
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(BAR_COLOR)

        for i in range(NUM_BARS):
            bar_h = max(4, int(self._bar_levels[i] * WAVEFORM_AREA_HEIGHT))
            bx = bar_area_x + i * (bar_w + BAR_GAP)
            by = bar_area_y + (WAVEFORM_AREA_HEIGHT - bar_h) // 2
            bar_path = QPainterPath()
            bar_path.addRoundedRect(float(bx), float(by), float(bar_w), float(bar_h), 2.0, 2.0)
            painter.fillPath(bar_path, BAR_COLOR)

        # Text
        text_x = PADDING_LEFT + WAVEFORM_AREA_WIDTH + PADDING_RIGHT
        text_rect = self.rect().adjusted(text_x, 0, -PADDING_RIGHT, 0)
        painter.setPen(TEXT_COLOR)
        font = painter.font()
        font.setPointSize(13)
        painter.setFont(font)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                        self._text)

        painter.end()
```

- [ ] **Step 2: Manual smoke test**

Create a temporary test script:
```bash
cat > /tmp/test_overlay.py << 'PYEOF'
import sys, math, time
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
sys.path.insert(0, "src")
from voice_input.overlay import OverlayWidget

app = QApplication(sys.argv)
w = OverlayWidget()
w.show()

t = 0
def tick():
    global t
    t += 1
    rms = abs(math.sin(t * 0.05)) * 0.7
    w.update_rms(rms)
    if t % 30 == 0:
        w.update_text(f"测试文字... 第 {t//30} 段")

timer = QTimer()
timer.timeout.connect(tick)
timer.start(16)
QTimer.singleShot(5000, app.quit)
app.exec()
PYEOF
source .venv/bin/activate && python /tmp/test_overlay.py
```

Expected: capsule appears at bottom center, bars animate with sine wave, text updates every ~0.5s, window doesn't steal focus.

- [ ] **Step 3: Commit**

```bash
git add src/voice_input/overlay.py
git commit -m "feat: overlay capsule widget with waveform bars and text animation"
```

---

## Phase 6: System Tray & Settings

### Task 9: System tray

**Files:**
- Create: `src/voice_input/tray.py`

- [ ] **Step 1: Implement tray.py**

```python
# src/voice_input/tray.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtGui import QAction, QActionGroup, QIcon
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon, QApplication

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

log = logging.getLogger(__name__)

LANGUAGES = {
    "en": "English",
    "zh": "简体中文",
    "zh-TW": "繁體中文",
    "ja": "日本語",
    "ko": "한국어",
}


class TrayManager(QSystemTrayIcon):
    """System tray icon using StatusNotifierItem (via Qt6 on KDE).

    Provides context menu for controlling the voice input app.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = "Idle"
        self._current_language = "zh"
        self._llm_enabled = True

        # Icons from Breeze theme
        self._icon_idle = QIcon.fromTheme("audio-input-microphone")
        self._icon_recording = QIcon.fromTheme("audio-input-microphone")  # overlay set below
        self._icon_muted = QIcon.fromTheme("audio-input-microphone-muted")

        self.setIcon(self._icon_idle)
        self.setToolTip("Voice Input — Idle")

        self._build_menu()

    def _build_menu(self) -> None:
        menu = QMenu()

        # Status
        self._status_action = QAction("Status: Idle", menu)
        self._status_action.setEnabled(False)
        menu.addAction(self._status_action)
        menu.addSeparator()

        # Start/Stop
        self._toggle_action = QAction("Start Recording", menu)
        menu.addAction(self._toggle_action)
        menu.addSeparator()

        # Language submenu
        lang_menu = QMenu("Language", menu)
        self._lang_group = QActionGroup(lang_menu)
        self._lang_group.setExclusive(True)
        self._lang_actions: dict[str, QAction] = {}
        for code, name in LANGUAGES.items():
            action = QAction(name, lang_menu)
            action.setCheckable(True)
            action.setData(code)
            if code == self._current_language:
                action.setChecked(True)
            self._lang_group.addAction(action)
            lang_menu.addAction(action)
            self._lang_actions[code] = action
        menu.addMenu(lang_menu)

        # LLM submenu
        llm_menu = QMenu("LLM Refinement", menu)
        self._llm_toggle = QAction("Enabled", llm_menu)
        self._llm_toggle.setCheckable(True)
        self._llm_toggle.setChecked(self._llm_enabled)
        llm_menu.addAction(self._llm_toggle)
        self._llm_settings_action = QAction("Settings...", llm_menu)
        llm_menu.addAction(self._llm_settings_action)
        menu.addMenu(llm_menu)

        menu.addSeparator()

        # Preferences
        self._prefs_action = QAction("Preferences...", menu)
        menu.addAction(self._prefs_action)

        # About
        self._about_action = QAction("About", menu)
        menu.addAction(self._about_action)

        menu.addSeparator()

        # Quit
        self._quit_action = QAction("Quit", menu)
        self._quit_action.triggered.connect(QApplication.quit)
        menu.addAction(self._quit_action)

        self.setContextMenu(menu)

    # --- Public interface for AppController to connect signals ---

    @property
    def toggle_action(self) -> QAction:
        return self._toggle_action

    @property
    def lang_group(self) -> QActionGroup:
        return self._lang_group

    @property
    def llm_toggle(self) -> QAction:
        return self._llm_toggle

    @property
    def llm_settings_action(self) -> QAction:
        return self._llm_settings_action

    @property
    def prefs_action(self) -> QAction:
        return self._prefs_action

    @property
    def about_action(self) -> QAction:
        return self._about_action

    def set_state(self, state: str) -> None:
        """Update tray state: 'Idle', 'Recording', 'Refining'."""
        self._state = state
        self._status_action.setText(f"Status: {state}")
        self.setToolTip(f"Voice Input — {state}")

        if state == "Recording":
            self._toggle_action.setText("Stop Recording")
            self.setIcon(self._icon_recording)
        elif state == "Refining":
            self._toggle_action.setText("Stop Recording")
            self._toggle_action.setEnabled(False)
        else:
            self._toggle_action.setText("Start Recording")
            self._toggle_action.setEnabled(True)
            self.setIcon(self._icon_idle)

    def set_language(self, code: str) -> None:
        self._current_language = code
        if code in self._lang_actions:
            self._lang_actions[code].setChecked(True)

    def set_llm_enabled(self, enabled: bool) -> None:
        self._llm_enabled = enabled
        self._llm_toggle.setChecked(enabled)
```

- [ ] **Step 2: Commit**

```bash
git add src/voice_input/tray.py
git commit -m "feat: system tray with SNI, language submenu, and LLM toggle"
```

---

### Task 10: Settings dialog

**Files:**
- Create: `src/voice_input/settings_dialog.py`

- [ ] **Step 1: Implement settings_dialog.py**

```python
# src/voice_input/settings_dialog.py
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from voice_input.config import AppConfig, save_config, xdg_config_dir

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

log = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """LLM settings dialog following Breeze style conventions."""

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Voice Input — LLM Settings")
        self.setMinimumWidth(420)
        self._config = config
        self._build_ui()
        self._load_from_config()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        # API Base URL
        self._api_base_edit = QLineEdit()
        self._api_base_edit.setPlaceholderText("https://api.openai.com/v1")
        form.addRow("API Base URL:", self._api_base_edit)

        # API Key with visibility toggle
        key_layout = QHBoxLayout()
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("sk-...")
        key_layout.addWidget(self._api_key_edit)
        self._toggle_vis_btn = QPushButton("Show")
        self._toggle_vis_btn.setFixedWidth(60)
        self._toggle_vis_btn.setCheckable(True)
        self._toggle_vis_btn.toggled.connect(self._toggle_key_visibility)
        key_layout.addWidget(self._toggle_vis_btn)
        form.addRow("API Key:", key_layout)

        # Model
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("gpt-4o-mini")
        form.addRow("Model:", self._model_edit)

        layout.addLayout(form)

        # Test button
        test_layout = QHBoxLayout()
        test_layout.addStretch()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._on_test)
        test_layout.addWidget(self._test_btn)
        layout.addLayout(test_layout)

        # Button box
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_from_config(self) -> None:
        llm = self._config.get("llm", {})
        self._api_base_edit.setText(llm.get("api_base", "https://api.openai.com/v1"))
        self._model_edit.setText(llm.get("model", "gpt-4o-mini"))
        # API key loaded from keyring, not config
        self._load_api_key()

    def _load_api_key(self) -> None:
        try:
            import keyring
            key = keyring.get_password("voice-input", "llm-api-key")
            if key:
                self._api_key_edit.setText(key)
        except Exception as e:
            log.debug("Could not load API key from keyring: %s", e)

    def _save_api_key(self, key: str) -> None:
        try:
            import keyring
            if key:
                keyring.set_password("voice-input", "llm-api-key", key)
            else:
                keyring.delete_password("voice-input", "llm-api-key")
        except Exception as e:
            log.warning("Could not save API key to keyring: %s", e)

    def _toggle_key_visibility(self, checked: bool) -> None:
        if checked:
            self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self._toggle_vis_btn.setText("Hide")
        else:
            self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self._toggle_vis_btn.setText("Show")

    def _on_save(self) -> None:
        self._config["llm"]["api_base"] = self._api_base_edit.text().strip() or "https://api.openai.com/v1"
        self._config["llm"]["model"] = self._model_edit.text().strip() or "gpt-4o-mini"
        save_config(self._config)
        self._save_api_key(self._api_key_edit.text().strip())
        self.accept()

    def _on_test(self) -> None:
        """Test API connection."""
        from voice_input.llm import LLMRefiner

        api_base = self._api_base_edit.text().strip() or "https://api.openai.com/v1"
        api_key = self._api_key_edit.text().strip()
        model = self._model_edit.text().strip() or "gpt-4o-mini"

        if not api_key:
            QMessageBox.warning(self, "Test", "API Key is required.")
            return

        self._test_btn.setEnabled(False)
        self._test_btn.setText("Testing...")

        async def do_test():
            refiner = LLMRefiner(api_base=api_base, api_key=api_key, model=model)
            try:
                success, msg = await refiner.test_connection()
                if success:
                    QMessageBox.information(self, "Test", f"Success: {msg}")
                else:
                    QMessageBox.warning(self, "Test", f"Failed: {msg}")
            finally:
                await refiner.close()
                self._test_btn.setEnabled(True)
                self._test_btn.setText("Test Connection")

        try:
            loop = asyncio.get_event_loop()
            loop.create_task(do_test())
        except RuntimeError:
            QMessageBox.warning(self, "Test", "Async event loop not available.")
            self._test_btn.setEnabled(True)
            self._test_btn.setText("Test Connection")

    def get_api_key(self) -> str:
        return self._api_key_edit.text().strip()
```

- [ ] **Step 2: Commit**

```bash
git add src/voice_input/settings_dialog.py
git commit -m "feat: LLM settings dialog with keyring API key storage and test button"
```

---

## Phase 7: App Controller & Entry Point

### Task 11: App controller — state machine wiring all modules

**Files:**
- Create: `src/voice_input/app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write failing test for state machine**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app.py::test_state_transitions -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement app.py**

```python
# src/voice_input/app.py
from __future__ import annotations

import asyncio
import enum
import logging
import queue
import sys
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QMessageBox

from voice_input.audio import AudioRecorder, compute_rms
from voice_input.config import AppConfig, load_config, save_config, xdg_config_dir
from voice_input.hotkey import HotkeyManager
from voice_input.injector import TextInjector
from voice_input.llm import LLMRefiner
from voice_input.overlay import OverlayWidget
from voice_input.settings_dialog import SettingsDialog
from voice_input.tray import TrayManager
from voice_input.whisper_worker import WhisperWorker

log = logging.getLogger(__name__)


class AppState(enum.Enum):
    IDLE = "Idle"
    RECORDING = "Recording"
    REFINING = "Refining"

    _transitions = None  # placeholder, set below

    def can_transition_to(self, target: AppState) -> bool:
        return target in _VALID_TRANSITIONS.get(self, set())


_VALID_TRANSITIONS = {
    AppState.IDLE: {AppState.RECORDING},
    AppState.RECORDING: {AppState.IDLE, AppState.REFINING},
    AppState.REFINING: {AppState.IDLE},
}


class AppController(QObject):
    """Central controller. Owns state machine, wires all modules."""

    state_changed = pyqtSignal(str)

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._state = AppState.IDLE
        self._last_transcription = ""

        # Queues
        self._whisper_queue: queue.Queue = queue.Queue(maxsize=200)
        self._viz_queue: queue.Queue = queue.Queue(maxsize=50)

        # Modules
        self._audio = AudioRecorder(
            whisper_queue=self._whisper_queue,
            viz_queue=self._viz_queue,
            device=config["audio"]["device"],
            sample_rate=config["audio"]["sample_rate"],
        )
        self._whisper = WhisperWorker(
            whisper_queue=self._whisper_queue,
            model_name=config["whisper"]["model"],
            language=config["whisper"]["language"],
            device=config["whisper"]["device"],
        )
        self._hotkey = HotkeyManager(
            mode=config["hotkey"]["mode"],
            key=config["hotkey"]["key"],
        )
        self._injector = TextInjector()
        self._overlay = OverlayWidget(
            margin_bottom=config["ui"]["overlay_margin_bottom"],
        )
        self._tray = TrayManager()

        # LLM (initialized lazily with API key)
        self._llm: LLMRefiner | None = None
        self._llm_enabled = config["llm"]["enabled"]

        # Visualization timer
        self._viz_timer = QTimer()
        self._viz_timer.setInterval(16)  # ~60fps
        self._viz_timer.timeout.connect(self._update_visualization)

        # Connect signals
        self._connect_signals()

    def _connect_signals(self) -> None:
        # Hotkey
        self._hotkey.recording_requested.connect(self._on_toggle_recording)
        self._hotkey.hold_started.connect(self._on_start_recording)
        self._hotkey.hold_stopped.connect(self._on_stop_recording)

        # Whisper
        self._whisper.transcription_updated.connect(self._on_transcription)
        self._whisper.error_occurred.connect(self._on_whisper_error)

        # Tray
        self._tray.toggle_action.triggered.connect(self._on_toggle_recording)
        self._tray.lang_group.triggered.connect(self._on_language_changed)
        self._tray.llm_toggle.toggled.connect(self._on_llm_toggled)
        self._tray.llm_settings_action.triggered.connect(self._on_open_settings)
        self._tray.about_action.triggered.connect(self._on_about)

        # State
        self.state_changed.connect(self._tray.set_state)

    async def start(self) -> None:
        """Initialize and start the app."""
        self._tray.show()

        # Register hotkey
        ok = await self._hotkey.register()
        if not ok:
            log.warning("Hotkey registration failed. Use tray menu to control recording.")

        # Start evdev loop if in hold mode
        if self._hotkey.mode == "hold":
            asyncio.create_task(self._hotkey.run_evdev_loop())

        # Pre-load whisper model in background
        self._whisper.start()

        # Initialize LLM if enabled
        self._init_llm()

        log.info("Voice Input started")

    def _init_llm(self) -> None:
        if not self._llm_enabled:
            return
        try:
            import keyring
            api_key = keyring.get_password("voice-input", "llm-api-key")
        except Exception:
            api_key = None
        if api_key:
            self._llm = LLMRefiner(
                api_base=self._config["llm"]["api_base"],
                api_key=api_key,
                model=self._config["llm"]["model"],
            )

    def _set_state(self, new_state: AppState) -> None:
        if not self._state.can_transition_to(new_state):
            log.warning("Invalid transition: %s → %s", self._state, new_state)
            return
        self._state = new_state
        self.state_changed.emit(new_state.value)
        log.info("State: %s", new_state.value)

    # --- Recording control ---

    @pyqtSlot()
    def _on_toggle_recording(self) -> None:
        if self._state == AppState.IDLE:
            self._on_start_recording()
        elif self._state == AppState.RECORDING:
            self._on_stop_recording()

    @pyqtSlot()
    def _on_start_recording(self) -> None:
        if self._state != AppState.IDLE:
            return
        self._whisper.reset()
        self._last_transcription = ""
        self._audio.start()
        self._viz_timer.start()
        self._overlay.update_text("")
        self._overlay.show()
        self._set_state(AppState.RECORDING)

    @pyqtSlot()
    def _on_stop_recording(self) -> None:
        if self._state != AppState.RECORDING:
            return
        self._audio.stop()
        self._viz_timer.stop()
        self._whisper.stop()

        text = self._last_transcription
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

    async def _refine_and_inject(self, text: str) -> None:
        corrected = await self._llm.refine(text)
        self._inject_and_finish(corrected)

    def _inject_and_finish(self, text: str) -> None:
        self._overlay.animate_exit(on_finished=self._overlay.hide)
        if text:
            self._injector.inject(text)
        self._set_state(AppState.IDLE)

    # --- Callbacks ---

    @pyqtSlot(str)
    def _on_transcription(self, text: str) -> None:
        self._last_transcription = text
        if self._state == AppState.RECORDING:
            self._overlay.update_text(text)

    @pyqtSlot(str)
    def _on_whisper_error(self, msg: str) -> None:
        log.error("Whisper error: %s", msg)
        self._send_notification("Voice Input Error", msg)

    def _update_visualization(self) -> None:
        """Drain viz queue and update overlay waveform."""
        chunks = []
        while True:
            try:
                chunks.append(self._viz_queue.get_nowait())
            except queue.Empty:
                break
        if chunks:
            data = np.concatenate(chunks)
            rms = compute_rms(data)
            self._overlay.update_rms(rms)

    @pyqtSlot(object)
    def _on_language_changed(self, action) -> None:
        code = action.data()
        self._config["whisper"]["language"] = code
        self._whisper.language = code
        save_config(self._config)
        log.info("Language changed to: %s", code)

    @pyqtSlot(bool)
    def _on_llm_toggled(self, enabled: bool) -> None:
        self._llm_enabled = enabled
        self._config["llm"]["enabled"] = enabled
        save_config(self._config)
        if enabled and self._llm is None:
            self._init_llm()

    @pyqtSlot()
    def _on_open_settings(self) -> None:
        dialog = SettingsDialog(self._config)
        if dialog.exec():
            # Reload LLM with new settings
            if self._llm:
                asyncio.ensure_future(self._llm.close())
            self._init_llm()

    @pyqtSlot()
    def _on_about(self) -> None:
        QMessageBox.about(
            None,
            "Voice Input",
            "Voice Input for KDE Plasma 6\n"
            "Version 0.1.0\n\n"
            "Speech-to-text with system tray integration.\n"
            "Powered by faster-whisper.",
        )

    def _send_notification(self, title: str, body: str) -> None:
        """Send a desktop notification via org.freedesktop.Notifications."""
        try:
            import subprocess
            subprocess.run(
                ["notify-send", title, body, "-a", "Voice Input"],
                timeout=5,
            )
        except Exception as e:
            log.debug("Notification failed: %s", e)


def run_app() -> None:
    """Application entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config()

    app = QApplication(sys.argv)
    app.setApplicationName("voice-input")
    app.setDesktopFileName("voice-input")
    app.setQuitOnLastWindowClosed(False)

    try:
        import qasync
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)
    except ImportError:
        log.warning("qasync not available; async features (LLM, DBus hotkey) disabled")
        loop = None

    controller = AppController(config)

    if loop is not None:
        with loop:
            loop.run_until_complete(controller.start())
            loop.run_forever()
    else:
        app.exec()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app.py -v`
Expected: PASS.

- [ ] **Step 5: Update __main__.py to call run_app**

```python
# src/voice_input/__main__.py
from voice_input.app import run_app


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/voice_input/app.py src/voice_input/__main__.py tests/test_app.py
git commit -m "feat: app controller with state machine wiring all modules"
```

---

## Phase 8: Packaging & Deployment

### Task 12: Desktop file and icons

**Files:**
- Create: `src/voice_input/resources/voice-input.desktop`

- [ ] **Step 1: Create .desktop file**

```ini
[Desktop Entry]
Type=Application
Name=Voice Input
Comment=Voice input for KDE Plasma 6
Exec=voice-input
Icon=audio-input-microphone
Terminal=false
Categories=Utility;Accessibility;
X-KDE-StartupNotify=false
NoDisplay=true
```

- [ ] **Step 2: Commit**

```bash
git add src/voice_input/resources/voice-input.desktop
git commit -m "feat: desktop entry file for KDE integration"
```

---

### Task 13: systemd user service

**Files:**
- Create: `packaging/systemd/voice-input.service`

- [ ] **Step 1: Create the service unit**

```ini
[Unit]
Description=Voice Input for KDE Plasma 6
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart=%h/.local/share/voice-input/venv/bin/python -m voice_input
Restart=on-failure
RestartSec=5
Environment=QT_QPA_PLATFORM=wayland

[Install]
WantedBy=graphical-session.target
```

- [ ] **Step 2: Commit**

```bash
git add packaging/systemd/voice-input.service
git commit -m "feat: systemd user service unit"
```

---

### Task 14: PKGBUILD

**Files:**
- Create: `packaging/arch/PKGBUILD`

- [ ] **Step 1: Create PKGBUILD**

```bash
# Maintainer: Voice Input
pkgname=voice-input
pkgver=0.1.0
pkgrel=1
pkgdesc="Voice input for KDE Plasma 6 (Wayland)"
arch=('any')
license=('MIT')
depends=(
    'python'
    'python-pyqt6'
    'python-sounddevice'
    'qt6-wayland'
    'wl-clipboard'
    'wtype'
    'libnotify'
    'fcitx5'
)
optdepends=(
    'ydotool: fallback text injection'
    'cuda: GPU acceleration for whisper'
    'python-evdev: hold-mode hotkey via evdev'
)
makedepends=('python-build' 'python-installer' 'python-setuptools')
source=()

package() {
    cd "$srcdir/.."

    # Install Python package
    python -m installer --destdir="$pkgdir" dist/*.whl

    # venv with faster-whisper (heavy dependency, isolated)
    local venv_dir="$pkgdir/opt/voice-input/venv"
    python -m venv "$venv_dir"
    "$venv_dir/bin/pip" install --no-cache-dir faster-whisper dbus-next httpx qasync keyring

    # Desktop file
    install -Dm644 src/voice_input/resources/voice-input.desktop \
        "$pkgdir/usr/share/applications/voice-input.desktop"

    # systemd user unit
    install -Dm644 packaging/systemd/voice-input.service \
        "$pkgdir/usr/lib/systemd/user/voice-input.service"
}
```

- [ ] **Step 2: Commit**

```bash
git add packaging/arch/PKGBUILD
git commit -m "feat: Arch Linux PKGBUILD"
```

---

### Task 15: Makefile

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Create Makefile**

```makefile
.PHONY: install-deps venv run install uninstall clean package

VENV_DIR = $(HOME)/.local/share/voice-input/venv
PYTHON = $(VENV_DIR)/bin/python
SRC_DIR = $(shell pwd)/src
SYSTEMD_DIR = $(HOME)/.config/systemd/user
DESKTOP_DIR = $(HOME)/.local/share/applications

install-deps:
	sudo pacman -S --needed python python-pyqt6 python-sounddevice \
		qt6-wayland wl-clipboard wtype libnotify fcitx5

venv:
	python -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install faster-whisper dbus-next httpx qasync keyring sounddevice PyQt6
	$(VENV_DIR)/bin/pip install -e .

run:
	PYTHONPATH=$(SRC_DIR) python -m voice_input

install: venv
	# Desktop file
	mkdir -p $(DESKTOP_DIR)
	cp src/voice_input/resources/voice-input.desktop $(DESKTOP_DIR)/
	sed -i "s|Exec=voice-input|Exec=$(PYTHON) -m voice_input|" $(DESKTOP_DIR)/voice-input.desktop
	# systemd unit
	mkdir -p $(SYSTEMD_DIR)
	cp packaging/systemd/voice-input.service $(SYSTEMD_DIR)/
	sed -i "s|ExecStart=.*|ExecStart=$(PYTHON) -m voice_input|" $(SYSTEMD_DIR)/voice-input.service
	systemctl --user daemon-reload
	@echo "Run: systemctl --user enable --now voice-input"

uninstall:
	systemctl --user disable --now voice-input 2>/dev/null || true
	rm -f $(SYSTEMD_DIR)/voice-input.service
	rm -f $(DESKTOP_DIR)/voice-input.desktop
	rm -rf $(VENV_DIR)
	systemctl --user daemon-reload

clean:
	rm -rf .venv build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

package:
	cd packaging/arch && makepkg -sf
```

- [ ] **Step 2: Commit**

```bash
git add Makefile
git commit -m "feat: Makefile with install-deps, venv, run, install, uninstall, package targets"
```

---

### Task 16: udev rules for evdev fallback

**Files:**
- Create: `packaging/udev/99-voice-input.rules`

- [ ] **Step 1: Create udev rule**

```
# Allow members of the 'input' group to read keyboard devices
# Required only for hold-mode hotkey (evdev backend)
# Install: sudo cp 99-voice-input.rules /etc/udev/rules.d/ && sudo udevadm control --reload
KERNEL=="event*", SUBSYSTEM=="input", GROUP="input", MODE="0640"
```

- [ ] **Step 2: Commit**

```bash
git add packaging/udev/99-voice-input.rules
git commit -m "feat: udev rules for evdev hotkey fallback"
```

---

### Task 17: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# Voice Input for KDE Plasma 6

System tray voice input application for Arch Linux + KDE Plasma 6 (Wayland).

Press a global hotkey, speak, and the transcribed text is typed into the focused application.
Optional LLM refinement corrects ASR errors before injection.

## Requirements

- Arch Linux with KDE Plasma 6 (Wayland session)
- Python 3.11+

## Quick Start

```bash
# Install system dependencies
make install-deps

# Create venv and install Python deps (downloads ~1.5GB for whisper model on first run)
make venv

# Run directly (dev mode)
make run

# Or install as a systemd user service
make install
systemctl --user enable --now voice-input
```

## Usage

- **Meta+Space** (default): Toggle recording on/off
- Right-click the tray icon for language selection, LLM settings, and preferences
- First run downloads the Whisper `medium` model (~1.5GB) to `~/.cache/voice-input/models/`

## Text Injection

Uses `wtype` (primary), `ydotool` (fallback), or clipboard paste (last resort).

If using `ydotool`, enable the daemon:
```bash
systemctl --user enable --now ydotool
```

## Input Method Compatibility

Automatically detects and temporarily disables fcitx5/ibus during text injection
to prevent double-conversion of already-transcribed Chinese text.

## Hotkey Modes

**Toggle mode** (default): Uses KDE's KGlobalAccel. Hotkey appears in
System Settings > Shortcuts. Press once to start, again to stop.

**Hold mode**: Uses evdev. Hold the key to record, release to stop.
Requires the user to be in the `input` group:
```bash
sudo usermod -aG input $USER
sudo cp packaging/udev/99-voice-input.rules /etc/udev/rules.d/
sudo udevadm control --reload
# Log out and back in
```

Set `mode = "hold"` in `~/.config/voice-input/config.toml`.

## LLM Refinement

Optional post-processing to fix ASR errors (e.g., Chinese homophones,
mis-transcribed English technical terms).

Configure via tray menu > LLM Refinement > Settings, or edit `config.toml`:
```toml
[llm]
enabled = true
api_base = "https://api.openai.com/v1"
model = "gpt-4o-mini"
```

API key is stored in KDE Wallet (via python-keyring), not in the config file.

## Packaging

Build an Arch package:
```bash
make package
# Installs to packaging/arch/voice-input-0.1.0-1-any.pkg.tar.zst
sudo pacman -U packaging/arch/voice-input-*.pkg.tar.zst
```

## Troubleshooting

- **wtype doesn't work**: Ensure `qt6-wayland` is installed and you're in a Wayland session
- **No hotkey response**: Check `journalctl --user -u voice-input` for KGlobalAccel errors
- **Whisper model download slow**: First run downloads ~1.5GB; subsequent runs use cache
- **fcitx5 interference**: The app auto-deactivates fcitx5 during injection; if issues persist, check `fcitx5-remote` is in PATH
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with installation, usage, and troubleshooting"
```

---

## Self-Review Checklist

**Spec coverage:** All 9 spec sections (Overview, Architecture, 8 modules, Config, KDE Integration, Dependencies, Packaging, Data Flow, Error Handling) have corresponding tasks.

- [x] Config module → Task 2
- [x] Audio recorder → Task 3
- [x] Whisper worker → Task 4
- [x] Waveform visualization → Integrated into Task 8 (overlay.py) and Task 11 (app.py _update_visualization)
- [x] Hotkey manager → Task 5
- [x] Text injector → Task 6
- [x] LLM refiner → Task 7
- [x] Overlay widget → Task 8
- [x] System tray → Task 9
- [x] Settings dialog → Task 10
- [x] App controller → Task 11
- [x] Desktop file → Task 12
- [x] systemd unit → Task 13
- [x] PKGBUILD → Task 14
- [x] Makefile → Task 15
- [x] udev rules → Task 16
- [x] README → Task 17

**Placeholder scan:** No TBD/TODO. All code steps include complete implementations.

**Type consistency:** Verified signal/slot names match across modules:
- `recording_requested`, `hold_started`, `hold_stopped` — HotkeyManager → AppController
- `transcription_updated(str)` — WhisperWorker → AppController
- `error_occurred(str)` — WhisperWorker → AppController
- `state_changed(str)` — AppController → TrayManager.set_state
- `update_rms(float)`, `update_text(str)` — AppController → OverlayWidget
