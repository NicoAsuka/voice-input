.PHONY: install-deps venv run install uninstall clean package test

VENV_DIR = $(HOME)/.local/share/voice-input/venv
PYTHON = $(VENV_DIR)/bin/python
SRC_DIR = $(shell pwd)/src
SYSTEMD_DIR = $(HOME)/.config/systemd/user
DESKTOP_DIR = $(HOME)/.local/share/applications

install-deps:
	sudo pacman -S --needed python python-pyqt6 python-sounddevice \
		qt6-wayland wl-clipboard ydotool libnotify python-evdev

venv:
	python -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -e ".[dev]"

test:
	python -m pytest tests/ -q

run:
	PYTHONPATH=$(SRC_DIR) python -m voice_input

install: venv
	mkdir -p $(DESKTOP_DIR)
	cp src/voice_input/resources/voice-input.desktop $(DESKTOP_DIR)/
	sed -i "s|Exec=voice-input|Exec=$(PYTHON) -m voice_input|" $(DESKTOP_DIR)/voice-input.desktop
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
