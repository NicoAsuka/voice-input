# src/voice_input/app.py
from __future__ import annotations

import asyncio
import enum
import logging
import queue
import sys

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMessageBox

from voice_input.audio import AudioRecorder, compute_rms
from voice_input.backends import create_backend
from voice_input.backends.base import RecognitionError, Session
from voice_input.backends.registry import BackendRegistry, RegistryState
from voice_input.config import AppConfig, load_config, save_config
from voice_input.hotkey import HotkeyManager
from voice_input.injector import TextInjector
from voice_input.overlay import OverlayWidget
from voice_input.postprocess.llm import LLMRefiner
from voice_input.postprocess.pipeline import ScenePipeline
from voice_input.postprocess.scene import SceneRegistry
from voice_input.settings_dialog import SettingsDialog
from voice_input.tray import TrayManager

log = logging.getLogger(__name__)

MIN_AUDIO_SAMPLES = 1600  # 0.1s at 16kHz


class AppState(enum.Enum):
    IDLE = "Idle"
    RECORDING = "Recording"
    TRANSCRIBING = "Transcribing"
    REFINING = "Refining"


_VALID_TRANSITIONS = {
    AppState.IDLE: {AppState.RECORDING},
    AppState.RECORDING: {AppState.IDLE, AppState.TRANSCRIBING},
    AppState.TRANSCRIBING: {AppState.IDLE, AppState.REFINING},
    AppState.REFINING: {AppState.IDLE},
}


def _can_transition(curr: AppState, target: AppState) -> bool:
    return target in _VALID_TRANSITIONS.get(curr, set())


class AppController(QObject):
    state_changed = pyqtSignal(str)

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._state = AppState.IDLE
        self._language = config.get("stt", {}).get("language", "zh")

        # Audio
        self._whisper_queue: queue.Queue = queue.Queue(maxsize=200)
        self._viz_queue: queue.Queue = queue.Queue(maxsize=50)
        self._audio = AudioRecorder(
            whisper_queue=self._whisper_queue,
            viz_queue=self._viz_queue,
            device=config["audio"]["device"],
            sample_rate=config["audio"]["sample_rate"],
        )

        self._registry = BackendRegistry(config, factory=create_backend)
        self._registry.add_state_listener(self._on_registry_state)
        self._current_session: Session | None = None

        # Postprocess
        self._scenes = SceneRegistry(config)
        self._llm: LLMRefiner | None = None
        self._llm_enabled = config.get("postprocess", {}).get("enabled", True)
        self._pipeline: ScenePipeline | None = None
        self._init_llm()

        # UI
        self._hotkey = HotkeyManager(
            mode=config["hotkey"]["mode"],
            key=config["hotkey"]["key"],
        )
        self._injector = TextInjector(
            paste_method=config.get("inject", {}).get("paste_method", "ctrl_v"),
        )
        self._overlay = OverlayWidget(
            margin_bottom=config["ui"]["overlay_margin_bottom"],
        )
        self._tray = TrayManager()
        self._tray.set_backend(config.get("stt", {}).get("backend", "sherpa"))
        self._tray.set_scenes(self._scenes.list(), active_id=self._scenes.active().id)

        # Visualization
        self._viz_timer = QTimer()
        self._viz_timer.setInterval(16)
        self._viz_timer.timeout.connect(self._update_visualization)

        self._connect_signals()

    def _init_llm(self) -> None:
        if not self._llm_enabled:
            self._llm = None
            self._pipeline = None
            return
        try:
            import keyring
            api_key = keyring.get_password("voice-input", "llm-api-key") or ""
        except Exception:
            api_key = ""
        self._llm = LLMRefiner(
            api_base=self._config["llm"]["api_base"],
            api_key=api_key,
            model=self._config["llm"]["model"],
        )
        self._pipeline = ScenePipeline(self._scenes, self._llm)

    def _connect_signals(self) -> None:
        self._hotkey.recording_requested.connect(self._on_toggle_recording)
        self._hotkey.hold_started.connect(self._on_start_recording)
        self._hotkey.hold_stopped.connect(self._on_stop_recording)
        self._tray.toggle_action.triggered.connect(self._on_toggle_recording)
        self._tray.lang_group.triggered.connect(self._on_language_changed)
        self._tray.stt_group.triggered.connect(self._on_backend_changed)
        self._tray.stt_settings_action.triggered.connect(self._on_open_settings)
        self._tray.scene_group.triggered.connect(self._on_scene_changed)
        self._tray.llm_toggle.toggled.connect(self._on_llm_toggled)
        self._tray.llm_settings_action.triggered.connect(self._on_open_settings)
        self._tray.about_action.triggered.connect(self._on_about)
        self.state_changed.connect(self._tray.set_state)

    async def start(self) -> None:
        self._tray.show()
        ok = await self._hotkey.register()
        if not ok:
            log.warning("Hotkey registration failed; use tray menu.")
        if self._hotkey.mode == "hold":
            asyncio.create_task(self._hotkey.run_evdev_loop())
        await self._registry.start()
        log.info(
            "Voice Input started (backend=%s)",
            self._config.get("stt", {}).get("backend", "sherpa"),
        )

    def _set_state(self, new_state: AppState) -> None:
        if not _can_transition(self._state, new_state):
            log.warning("Invalid transition: %s -> %s", self._state, new_state)
            return
        self._state = new_state
        self.state_changed.emit(new_state.value)

    def _on_registry_state(self, state: RegistryState, error: str | None) -> None:
        log.info("registry state -> %s (error=%s)", state.value, error)
        self._tray.set_backend_status(state.value, error)
        if state == RegistryState.ERROR and error:
            self._send_notification("Voice Input", f"Backend error: {error[:80]}")

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
        if not self._registry.is_ready():
            state = self._registry.state()
            if state == RegistryState.LOADING:
                self._send_notification("Voice Input", "Model loading, please wait")
            elif state == RegistryState.ERROR:
                err = self._registry.last_error() or "unknown error"
                self._send_notification("Voice Input", f"Backend error: {err[:80]}")
            else:
                self._send_notification("Voice Input", "Backend not ready")
            return

        try:
            self._current_session = self._registry.create_session(self._language)
        except Exception as e:
            log.exception("create_session failed")
            self._send_notification("Voice Input", f"Cannot create session: {e}")
            return

        self._drain_audio_queue()
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

        audio_buffer = self._drain_audio_queue()
        log.info("Audio buffer: %d samples (min=%d)", len(audio_buffer), MIN_AUDIO_SAMPLES)
        if self._current_session is not None and len(audio_buffer) > 0:
            self._current_session.push_audio(audio_buffer)

        if len(audio_buffer) < MIN_AUDIO_SAMPLES:
            if self._current_session:
                self._current_session.cancel()
            self._current_session = None
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return

        self._set_state(AppState.TRANSCRIBING)
        self._overlay.update_text("Transcribing...")
        log.info("Starting transcription...")
        asyncio.ensure_future(self._finish_and_inject())

    def _drain_audio_queue(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._whisper_queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.array([], dtype=np.int16)
        return np.concatenate(chunks)

    async def _finish_and_inject(self) -> None:
        log.info("_finish_and_inject called")
        session = self._current_session
        self._current_session = None
        if session is None:
            log.warning("_finish_and_inject: session is None")
            self._set_state(AppState.IDLE)
            return

        try:
            log.info("Calling session.finish()...")
            text = await session.finish()
        except RecognitionError as e:
            log.warning("recognition error: %s", e)
            self._send_notification("Voice Input", e.user_message)
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return
        except Exception as e:
            log.exception("session.finish unexpected error")
            self._send_notification("Voice Input", f"Recognition failed: {str(e)[:80]}")
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return

        if not text:
            self._send_notification("Voice Input", "No speech detected")
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return

        if self._pipeline is not None:
            self._set_state(AppState.REFINING)
            self._overlay.update_text("Refining...")
            try:
                text = await self._pipeline.process(text)
            except Exception:
                log.exception("postprocess failed; using raw text")

        self._inject_and_finish(text)

    def _inject_and_finish(self, text: str) -> None:
        self._set_state(AppState.IDLE)
        if not text:
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            return

        def _do_inject() -> None:
            import threading
            threading.Thread(
                target=self._injector.inject, args=(text,), daemon=True
            ).start()

        self._overlay.animate_exit(
            on_finished=lambda: (self._overlay.hide(), _do_inject())
        )

    def _update_visualization(self) -> None:
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._viz_queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return
        data = np.concatenate(chunks)
        rms = compute_rms(data)
        self._overlay.update_rms(rms)

    @pyqtSlot(QAction)
    def _on_language_changed(self, action: QAction) -> None:
        code = action.data()
        self._config.setdefault("stt", {})["language"] = code
        self._language = code
        save_config(self._config)
        log.info("Language -> %s", code)

    @pyqtSlot(QAction)
    def _on_backend_changed(self, action: QAction) -> None:
        backend_name = action.data()
        if backend_name == self._config.get("stt", {}).get("backend"):
            return
        self._config.setdefault("stt", {})["backend"] = backend_name
        save_config(self._config)
        asyncio.ensure_future(self._registry.synchronize(self._config))
        log.info("Backend -> %s (reload scheduled)", backend_name)

    @pyqtSlot(QAction)
    def _on_scene_changed(self, action: QAction) -> None:
        scene_id = action.data()
        try:
            self._scenes.set_active(scene_id)
        except KeyError:
            log.warning("Unknown scene: %s", scene_id)
            return
        self._config.setdefault("postprocess", {})["active_scene"] = scene_id
        save_config(self._config)
        log.info("Scene -> %s", scene_id)

    @pyqtSlot(bool)
    def _on_llm_toggled(self, enabled: bool) -> None:
        self._llm_enabled = enabled
        self._config.setdefault("postprocess", {})["enabled"] = enabled
        save_config(self._config)
        if enabled and self._llm is None:
            self._init_llm()
        elif not enabled:
            if self._llm:
                asyncio.ensure_future(self._llm.close())
            self._llm = None
            self._pipeline = None

    @pyqtSlot()
    def _on_open_settings(self) -> None:
        dialog = SettingsDialog(self._config)
        if dialog.exec():
            if self._llm:
                asyncio.ensure_future(self._llm.close())
            self._init_llm()
            asyncio.ensure_future(self._registry.synchronize(self._config))

    @pyqtSlot()
    def _on_about(self) -> None:
        QMessageBox.about(
            None, "Voice Input",
            "Voice Input for KDE Plasma 6\nVersion 0.2.0\n\nSpeech-to-text via sherpa-onnx.",
        )

    def _send_notification(self, title: str, body: str) -> None:
        try:
            import subprocess
            subprocess.run(
                ["notify-send", title, body, "-a", "Voice Input"], timeout=5,
            )
        except Exception as e:
            log.debug("Notification failed: %s", e)


def run_app() -> None:
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
        log.warning("qasync not available")
        loop = None

    controller = AppController(config)
    if loop is not None:
        with loop:
            loop.run_until_complete(controller.start())
            loop.run_forever()
    else:
        app.exec()
