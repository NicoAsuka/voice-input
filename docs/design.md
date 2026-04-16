# Voice Input for KDE Plasma 6 (Wayland) - Design Spec

**Date:** 2026-04-17
**Location:** `/home/Nico/Project/voice-input/`
**Status:** Approved

## 1. Overview

A system tray voice input application for Arch Linux + KDE Plasma 6 + Wayland. The user presses a global hotkey, speaks, and the transcribed text is injected into the focused application. An optional LLM refinement step corrects ASR errors before injection.

**Hard constraints:**
- KDE Plasma 6 + Wayland only (no X11, no GNOME)
- Python 3.11+ with PyQt6
- Qt6 / KDE Frameworks 6 native integration

## 2. Architecture

Single-process, multi-thread + asyncio hybrid.

```
┌─────────────────────────────────────────────────┐
│                Main Thread (Qt6 Event Loop)       │
│  ┌───────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ SystemTray│ │ Overlay  │ │ Settings Dialog│  │
│  │   (SNI)   │ │(LayerShl)│ │   (QDialog)    │  │
│  └───────────┘ └──────────┘ └────────────────┘  │
│         │            ▲              ▲             │
│         ▼            │              │             │
│  ┌──────────────────────────────────────────┐    │
│  │          AppController (QObject)          │    │
│  │  State: Idle → Recording → Refining       │    │
│  └──────┬───────────┬───────────┬───────────┘    │
│         │           │           │                 │
│  ┌──────▼──┐ ┌──────▼──┐ ┌─────▼──────┐         │
│  │Hotkey   │ │Audio    │ │TextInjector│         │
│  │Manager  │ │Recorder │ │(wtype/ydo) │         │
│  │(DBus)   │ │(Thread) │ └────────────┘         │
│  └─────────┘ └────┬────┘                         │
│                    │                              │
│              ┌─────▼─────┐  ┌──────────────┐     │
│              │ Whisper    │  │ LLM Refiner  │     │
│              │ (QThread)  │  │ (asyncio)    │     │
│              └────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────┘
```

**AppController** is the central coordinator. It owns a state machine (Idle/Recording/Refining) and connects all modules via Qt signals/slots. All UI updates are dispatched back to the main thread via signals.

## 3. Modules

### 3.1 Global Hotkey (HotkeyManager)

**Primary: KGlobalAccel via DBus (toggle mode)**
- Connect to session bus with `dbus-next` (async)
- Register component `voice-input`, action `toggle-recording` on `org.kde.kglobalaccel` at `/kglobalaccel`
- Default key: `Meta+Space`
- Toggle semantics: first press starts recording, second press stops
- Appears in KDE System Settings > Shortcuts, user-rebindable

**Alternative: evdev (hold mode)**
- Use `python-evdev` to read `/dev/input/event*`
- True press-and-hold: key down starts recording, key up stops
- Requires user in `input` group; provide udev rule `99-voice-input.rules`
- No KDE Settings integration

**Config:** `hotkey_mode = "toggle"` (default) or `"hold"` in config.toml.

**Fallback:** If KGlobalAccel DBus registration fails at startup, log warning and prompt user to switch to evdev mode or use manual tray menu trigger.

### 3.2 Audio Recorder (AudioRecorder)

- `sounddevice.InputStream`, 16kHz mono, int16 format
- The sounddevice callback runs in its own thread; it pushes frames into two `queue.Queue` instances:
  - `whisper_queue` — consumed by WhisperWorker
  - `viz_queue` — consumed by main thread for waveform visualization
- Start/stop controlled by AppController signals

### 3.3 Waveform Visualization

- Main thread `QTimer` fires every 16ms (~60fps)
- Reads all available frames from `viz_queue`
- Computes RMS over 256-sample windows
- Drives 5 vertical bars in the overlay capsule:
  - Bar weights: `[0.5, 0.8, 1.0, 0.75, 0.55]`
  - Smoothing envelope: attack coefficient 0.4, release coefficient 0.15
  - Per-bar random jitter: ±4%
- Rendered with `QPainter` inside the overlay widget's `paintEvent`

### 3.4 Speech Recognition (WhisperWorker)

- Runs in a `QThread`
- Every 0.5s, drains `whisper_queue` and appends to a cumulative audio buffer
- Feeds the full buffer to `faster-whisper` for transcription
- Emits `transcription_updated(str)` signal to main thread
- Model: `medium` by default, stored at `~/.cache/voice-input/models/`
- Device: try `cuda` first, fall back to `cpu` with `int8` quantization
- Auto-downloads model on first use
- Default language: `zh`; switchable to `en`, `zh-TW`, `ja`, `ko` via tray menu

### 3.5 Overlay Widget (OverlayWidget)

**Wayland Layer Shell (primary):**
- Use `zwlr_layer_shell_v1` protocol via ctypes + `libwayland-client`
- Layer: `overlay`, Anchor: `bottom`, Margin: 80px from bottom
- `keyboard_interactivity: none` — never steals focus

**Fallback:**
- `Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool`
- `setAttribute(Qt.WA_ShowWithoutActivating)`

**KWin Blur:**
- Use `org_kde_kwin_blur_manager` Wayland protocol to enable blur behind
- Background: `rgba(30, 30, 30, 0.65)`
- Fallback (no blur): `rgba(30, 30, 30, 0.92)` solid

**Capsule appearance:**
- Height: 56px, border-radius: 28px
- No border, no title bar
- Left side: 5-bar waveform in 44x32px area
- Right side: text label, elastic width 160–560px, real-time transcription
- Capsule width expands with `QPropertyAnimation`, duration 0.25s

**Animations:**
- Entry: 0.35s, `QEasingCurve.OutBack` (spring feel)
- Exit: 0.22s, scale to 0.9 + fade out

**Font:** System font (Noto Sans CJK, default on Arch + KDE)

### 3.6 Text Injection (TextInjector)

**Priority chain:**
1. `wtype` — virtual keyboard protocol, Plasma 6 supports it
2. `ydotool` — requires `ydotoold` systemd user service
3. Clipboard fallback — `wl-copy` + `wtype -M ctrl v` or `ydotool key` for Ctrl+V

**Input method handling:**
- Before injection: query `fcitx5-remote -s` for current IM state
- If CJK IM active: `fcitx5-remote -c` to deactivate
- After injection: `fcitx5-remote -o` to restore
- Also handle `ibus` via `ibus engine` query/switch

**Clipboard protection:**
- Before injection: save `wl-paste` content (text/plain and x-kde-passwordManagerHint)
- After injection: restore with `wl-copy`

### 3.7 LLM Refinement (LLMRefiner)

- OpenAI-compatible API via `httpx.AsyncClient`
- Bridged to Qt event loop with `qasync`
- Configurable: API base URL, API key, model name
- System prompt (strict conservative correction only — see spec)
- Timeout: 5s; on failure/timeout, fall back to raw transcription
- API key stored in KDE Wallet via `python-keyring` with kwallet backend
- Enabled/disabled via tray menu toggle

### 3.8 System Tray (TrayManager)

- `QSystemTrayIcon` (Qt6 auto-uses StatusNotifierItem on KDE)
- Icon: `audio-input-microphone` (Breeze theme), switches to red-dot overlay during recording
- Context menu:
  - Status indicator (Idle / Recording / Refining)
  - Start/Stop Recording (manual trigger)
  - Language → submenu (English / 简体中文 / 繁體中文 / 日本語 / 한국어), single-select
  - LLM Refinement → submenu (Enable/Disable toggle, Settings...)
  - Preferences... (hotkey, model, audio device)
  - About
  - Quit

### 3.9 Settings Dialog

- `QDialog`, Breeze-themed
- Fields:
  - API Base URL (default `https://api.openai.com/v1`)
  - API Key (password echo mode, toggle-visibility button)
  - Model (default `gpt-4o-mini`)
  - [Test] button — sends ping, shows result via KNotification or QMessageBox
  - [Save] / [Cancel] buttons
- Saves to `~/.config/voice-input/config.toml`

## 4. Configuration & Storage

Following XDG Base Directory:
- Config: `~/.config/voice-input/config.toml`
- Cache: `~/.cache/voice-input/` (whisper models)
- Data: `~/.local/share/voice-input/`

**config.toml schema:**
```toml
[hotkey]
mode = "toggle"          # "toggle" or "hold"
key = "Meta+Space"

[whisper]
model = "medium"
language = "zh"
device = "auto"          # "auto", "cuda", "cpu"

[llm]
enabled = true
api_base = "https://api.openai.com/v1"
model = "gpt-4o-mini"
# api_key stored in KDE Wallet, not here

[audio]
device = "default"
sample_rate = 16000

[ui]
overlay_margin_bottom = 80
```

## 5. KDE Integration

- Notifications: KNotification (via DBus `org.freedesktop.Notifications`) for errors and state changes
- `.desktop` file: `X-KDE-StartupNotify=false`, `NoDisplay=true`
- Icons: Breeze theme icon names (`audio-input-microphone`, `audio-input-microphone-muted`)

## 6. Dependencies

**System (pacman):**
- python, python-pyqt6, qt6-wayland, python-sounddevice
- wl-clipboard, wtype, libnotify, fcitx5

**Optional (pacman/AUR):**
- ydotool (fallback injector)
- cuda (GPU whisper)

**Python (pip in venv):**
- faster-whisper, dbus-next, httpx, qasync, toml/tomli, keyring

**Venv location:** `~/.local/share/voice-input/venv/` (user install) or `/opt/voice-input/venv/` (system PKGBUILD)

## 7. Packaging

**Project structure:** `pyproject.toml` (PEP 621) + src layout

```
voice-input/
├── src/voice_input/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py              # AppController, QApplication setup
│   ├── hotkey.py            # HotkeyManager
│   ├── audio.py             # AudioRecorder
│   ├── whisper_worker.py    # WhisperWorker
│   ├── overlay.py           # OverlayWidget + waveform
│   ├── injector.py          # TextInjector
│   ├── llm.py               # LLMRefiner
│   ├── tray.py              # TrayManager
│   ├── settings_dialog.py   # Settings QDialog
│   ├── config.py            # Config load/save
│   └── resources/
│       ├── icons/
│       └── voice-input.desktop
├── packaging/
│   ├── arch/PKGBUILD
│   └── systemd/voice-input.service
├── Makefile
├── pyproject.toml
└── README.md
```

**Makefile targets:** `install-deps`, `venv`, `run`, `install`, `uninstall`, `clean`, `package`

**systemd user unit:** `voice-input.service` with `Type=simple`, `ExecStart` pointing to venv python + module

## 8. Data Flow (Happy Path)

1. User presses Meta+Space → HotkeyManager emits `recording_requested`
2. AppController transitions Idle → Recording, starts AudioRecorder, shows OverlayWidget
3. AudioRecorder feeds chunks to whisper_queue and viz_queue
4. WhisperWorker transcribes, emits `transcription_updated` → overlay text updates in real-time
5. User presses Meta+Space again → HotkeyManager emits `recording_requested`
6. AppController stops AudioRecorder, transitions Recording → Refining
7. If LLM enabled: LLMRefiner corrects text (5s timeout), overlay shows "refining..." state
8. TextInjector handles fcitx5, injects text via wtype, restores clipboard
9. OverlayWidget plays exit animation, AppController transitions → Idle

## 9. Error Handling

- **Whisper model download fails:** KNotification error, tray stays in Idle, user can retry
- **No microphone:** KNotification on startup, recording disabled, tray menu grayed out
- **wtype unavailable:** try ydotool, then clipboard fallback; log which method is active
- **LLM timeout/failure:** use raw transcription, no error shown to user (silent fallback)
- **KGlobalAccel registration fails:** log warning, suggest evdev mode or manual tray trigger
- **Layer Shell unavailable:** use Qt fallback window flags, log at debug level
