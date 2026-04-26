# src/voice_input/app.py
from __future__ import annotations

import asyncio
import enum
import importlib.util
import logging
import queue
import sys

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMessageBox

from voice_input.audio import AudioRecorder, compute_rms
from voice_input.backends import create_backend
from voice_input.config import AppConfig, load_config, save_config
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
        self._language = config.get("stt", {}).get("local", {}).get("language", "zh")
        self._backend = create_backend(config)
        self._backend_is_streaming = self._backend.is_streaming()
        self._worker_ready = False
        self._whisper = self._create_worker()
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
        self._tray.set_backend(config.get("stt", {}).get("backend", "local"))
        self._tray.set_engine(
            config.get("stt", {}).get("local", {}).get("engine", "whisper")
        )

        # LLM (initialized lazily with API key)
        self._llm: LLMRefiner | None = None
        self._llm_enabled = config["llm"]["enabled"]

        # Visualization timer
        self._viz_timer = QTimer()
        self._viz_timer.setInterval(16)  # ~60fps
        self._viz_timer.timeout.connect(self._update_visualization)

        # Connect signals
        self._connect_signals()

    def _create_worker(self) -> WhisperWorker:
        worker = WhisperWorker(
            whisper_queue=self._whisper_queue,
            backend=self._backend,
            language=self._language,
        )
        worker.model_ready.connect(self._on_worker_ready)
        worker.transcription_updated.connect(self._on_transcription)
        worker.error_occurred.connect(self._on_whisper_error)
        return worker

    def _connect_signals(self) -> None:
        # Hotkey
        self._hotkey.recording_requested.connect(self._on_toggle_recording)
        self._hotkey.hold_started.connect(self._on_start_recording)
        self._hotkey.hold_stopped.connect(self._on_stop_recording)

        # Tray
        self._tray.toggle_action.triggered.connect(self._on_toggle_recording)
        self._tray.lang_group.triggered.connect(self._on_language_changed)
        self._tray.stt_group.triggered.connect(self._on_backend_changed)
        self._tray.stt_settings_action.triggered.connect(self._on_open_settings)
        self._tray.engine_group.triggered.connect(self._on_engine_changed)
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

        # Start worker for both streaming and non-streaming backends. Streaming
        # backends emit partial text; non-streaming backends only buffer audio.
        self._whisper.start()

        # Initialize LLM if enabled
        self._init_llm()

        log.info(
            "Voice Input started (backend=%s)",
            self._config.get("stt", {}).get("backend", "local"),
        )

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
            log.warning("Invalid transition: %s -> %s", self._state, new_state)
            return
        self._state = new_state
        self.state_changed.emit(new_state.value)
        log.info("State: %s", new_state.value)

    # --- Recording control ---

    @pyqtSlot()
    def _on_toggle_recording(self) -> None:
        log.info("Toggle recording requested, current state: %s", self._state.value)
        if self._state == AppState.IDLE:
            self._on_start_recording()
        elif self._state == AppState.RECORDING:
            self._on_stop_recording()
        else:
            log.warning("Ignoring toggle in state %s", self._state.value)

    @pyqtSlot()
    def _on_start_recording(self) -> None:
        if self._state != AppState.IDLE:
            return
        self._whisper.reset()
        self._whisper.set_active(True)
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
        self._whisper.set_active(False)
        audio_buffer = self._whisper.get_audio_buffer()

        if self._backend_is_streaming:
            if len(audio_buffer) < 1600 and not self._last_transcription:
                self._overlay.animate_exit(on_finished=self._overlay.hide)
                self._set_state(AppState.IDLE)
                return

            self._set_state(AppState.TRANSCRIBING)
            self._overlay.update_text("识别中...")
            asyncio.ensure_future(
                self._transcribe_and_inject(
                    audio_buffer,
                    fallback_text=self._last_transcription,
                )
            )
        else:
            if len(audio_buffer) < 1600:
                self._overlay.animate_exit(on_finished=self._overlay.hide)
                self._set_state(AppState.IDLE)
                return
            self._set_state(AppState.TRANSCRIBING)
            self._overlay.update_text("识别中...")
            asyncio.ensure_future(self._transcribe_and_inject(audio_buffer))

    async def _refine_and_inject(self, text: str) -> None:
        corrected = await self._llm.refine(text)
        self._inject_and_finish(corrected)

    async def _transcribe_and_inject(
        self,
        audio_buffer: np.ndarray,
        fallback_text: str = "",
    ) -> None:
        """Run final STT on the full audio buffer, then optionally refine and inject."""
        try:
            text = await self._backend.transcribe(audio_buffer, self._language)
        except Exception as e:
            log.error("Transcription failed: %s", e)
            self._send_notification("Voice Input Error", f"STT API error: {e}")
            text = ""

        if not text and fallback_text:
            log.warning("Final transcription returned empty; using last partial text")
            text = fallback_text

        if not text:
            self._overlay.animate_exit(on_finished=self._overlay.hide)
            self._set_state(AppState.IDLE)
            return

        self._last_transcription = text
        if self._llm_enabled and self._llm is not None:
            self._set_state(AppState.REFINING)
            self._overlay.update_text("Refining...")
            await self._refine_and_inject(text)
        else:
            self._inject_and_finish(text)

    def _inject_and_finish(self, text: str) -> None:
        self._set_state(AppState.IDLE)
        if text:
            # Start injection after overlay finishes its exit animation,
            # otherwise ydotool's Ctrl+V may be sent to the overlay instead
            # of the target application.
            def _do_inject() -> None:
                import threading

                threading.Thread(
                    target=self._injector.inject,
                    args=(text,),
                    daemon=True,
                ).start()

            self._overlay.animate_exit(
                on_finished=lambda: (self._overlay.hide(), _do_inject())
            )
        else:
            self._overlay.animate_exit(on_finished=self._overlay.hide)

    # --- Callbacks ---

    @pyqtSlot(str)
    def _on_transcription(self, text: str) -> None:
        self._last_transcription = text
        if self._state == AppState.RECORDING and self._backend_is_streaming:
            self._overlay.update_text(text)

    @pyqtSlot(str)
    def _on_whisper_error(self, msg: str) -> None:
        log.error("Whisper error: %s", msg)
        self._send_notification("Voice Input Error", msg)

    @pyqtSlot()
    def _on_worker_ready(self) -> None:
        self._worker_ready = True

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

    @pyqtSlot(QAction)
    def _on_language_changed(self, action: QAction) -> None:
        code = action.data()
        self._config["stt"]["local"]["language"] = code
        self._language = code
        self._whisper.language = code
        save_config(self._config)
        log.info("Language changed to: %s", code)

    def _can_restart_backend_worker(self) -> bool:
        if getattr(self, "_state", AppState.IDLE) != AppState.IDLE:
            self._send_notification(
                "Voice Input",
                "Cannot switch STT backend while recording or transcribing.",
            )
            return False
        if not getattr(self, "_worker_ready", False):
            self._send_notification(
                "Voice Input",
                "STT backend is still starting. Try again after it is ready.",
            )
            return False
        return True

    def _restart_backend_worker(self) -> bool:
        old_backend = self._backend
        if self._whisper.isRunning():
            self._whisper.stop()
            if not self._whisper.wait(2000):
                self._send_notification(
                    "Voice Input",
                    "Timed out stopping current STT backend.",
                )
                return False

        asyncio.ensure_future(old_backend.cleanup())
        self._backend = create_backend(self._config)
        self._backend_is_streaming = self._backend.is_streaming()
        self._worker_ready = False
        self._whisper = self._create_worker()
        self._whisper.start()
        return True

    @pyqtSlot(QAction)
    def _on_backend_changed(self, action: QAction) -> None:
        backend_name = action.data()
        if backend_name == self._config.get("stt", {}).get("backend"):
            return
        if not self._can_restart_backend_worker():
            self._tray.set_backend(self._config.get("stt", {}).get("backend", "local"))
            return

        old_backend_name = self._config.get("stt", {}).get("backend", "local")
        self._config.setdefault("stt", {})["backend"] = backend_name
        if self._restart_backend_worker():
            save_config(self._config)
            log.info("STT backend changed to: %s", backend_name)
        else:
            self._config.setdefault("stt", {})["backend"] = old_backend_name
            self._tray.set_backend(old_backend_name)

    @pyqtSlot(QAction)
    def _on_engine_changed(self, action: QAction) -> None:
        engine = action.data()
        local_cfg = self._config.setdefault("stt", {}).setdefault("local", {})
        if engine == local_cfg.get("engine"):
            return
        if engine == "sensevoice" and importlib.util.find_spec("funasr") is None:
            self._send_notification(
                "Voice Input",
                "SenseVoice requires funasr. Install voice-input[sensevoice] first.",
            )
            self._tray.set_engine(local_cfg.get("engine", "whisper"))
            return
        if not self._can_restart_backend_worker():
            self._tray.set_engine(local_cfg.get("engine", "whisper"))
            return

        old_engine = local_cfg.get("engine", "whisper")
        old_model = local_cfg.get("model", "")
        local_cfg["engine"] = engine
        if engine == "sensevoice" and local_cfg.get("model") in {
            "",
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
        }:
            local_cfg["model"] = "iic/SenseVoiceSmall"

        if self._config.get("stt", {}).get("backend", "local") == "local":
            if not self._restart_backend_worker():
                local_cfg["engine"] = old_engine
                local_cfg["model"] = old_model
                self._tray.set_engine(old_engine)
                return
        save_config(self._config)
        log.info("Local engine changed to: %s", engine)

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
            "Powered by faster-whisper and SenseVoice.",
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
