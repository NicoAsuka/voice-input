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
