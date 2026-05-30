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

# Create venv and install Python deps
make venv

# Run directly (dev mode)
make run

# Or install as a systemd user service
make install
systemctl --user enable --now voice-input
```

On first run the app downloads the sherpa-onnx speech model (~237 MB) to
`~/.cache/voice-input/sherpa-models/`.

## Usage

- **Meta+Space** (default): Toggle recording on/off
- Right-click the tray icon for language selection, scene selection, LLM settings, and preferences

## Text Injection

Uses `wl-copy` (clipboard) + `ydotool` (key simulation) to paste text into the focused application.

Ensure the ydotool daemon is running:
```bash
systemctl --user enable --now ydotool
```

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

## STT Backends

- **sherpa-onnx** (default): Local offline recognition using sherpa-onnx models. No network required.
- **OpenAI Whisper API**: Cloud-based, requires API key.
- **Google Speech-to-Text**: Cloud-based, requires service account credentials.
- **Volcengine Speech**: Cloud-based (字节火山), requires app ID and access key.

Switch via tray menu > STT Backend, or edit `config.toml`.

## LLM Refinement

Optional post-processing to fix ASR errors (e.g., Chinese homophones,
mis-transcribed English technical terms).

Configure via tray menu > LLM Refinement > Settings, or edit `config.toml`:
```toml
[llm]
api_base = "https://api.openai.com/v1"
model = "gpt-4o-mini"
```

API key is stored in KDE Wallet (via python-keyring), not in the config file.

## Packaging

Build an Arch package:
```bash
make package
sudo pacman -U packaging/arch/voice-input-*.pkg.tar.zst
```

## Troubleshooting

- **Text injection doesn't work**: Ensure `wl-clipboard` and `ydotool` are installed, and `ydotool` daemon is running (`systemctl --user status ydotool`)
- **No hotkey response**: Check `journalctl --user -u voice-input` for KGlobalAccel errors
- **Model download slow or fails**: First run downloads ~237 MB; check network and `~/.cache/voice-input/sherpa-models/`
- **No speech detected**: Check microphone is working (`arecord -d 3 test.wav && aplay test.wav`)
