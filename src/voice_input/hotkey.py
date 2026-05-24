# src/voice_input/hotkey.py
from __future__ import annotations

import asyncio
import logging

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
        self._dbus_bus = None  # prevent GC of DBus connection
        self._dbus_proxy = None
        self._dbus_comp_proxy = None
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
        """Register via KGlobalAccel DBus interface.

        KGlobalAccel API (Plasma 5/6):
          doRegister(actionId: as)
          setShortcut(actionId: as, keys: ai, flags: u) -> ai
        keys is an array of Qt key+modifier integers.
        """
        try:
            from dbus_next.aio import MessageBus
            from PyQt6.QtGui import QKeySequence

            bus = await MessageBus().connect()
            self._dbus_bus = bus  # prevent GC
            introspection = await bus.introspect(
                "org.kde.kglobalaccel", "/kglobalaccel"
            )
            proxy = bus.get_proxy_object(
                "org.kde.kglobalaccel", "/kglobalaccel", introspection
            )
            self._dbus_proxy = proxy  # prevent GC
            iface = proxy.get_interface("org.kde.KGlobalAccel")

            # action_id: [componentUnique, actionUnique, actionFriendlyName, componentFriendlyName]
            action_id = ["voice-input", "toggle-recording", "Toggle Recording", "Voice Input"]

            # Step 1: register the action
            await iface.call_do_register(action_id)

            # Step 2: parse key string to Qt integer code
            seq = QKeySequence(self.key_string)
            if seq.isEmpty():
                log.warning("Could not parse key sequence: %s", self.key_string)
                return False
            key_combo = seq[0]
            # toCombined() → int (Qt6), or directly int (Qt5)
            key_int = key_combo.toCombined() if hasattr(key_combo, "toCombined") else int(key_combo)

            # Step 3: set the shortcut (flags: 0x2 = SetPresent)
            await iface.call_set_shortcut(action_id, [key_int], 2)

            # Step 4: listen for component shortcut trigger via globalShortcutsByKey polling
            # or subscribe to yourShortcutGotChanged (fires on key-change, not on press).
            # For press events, subscribe to the component object signal.
            await self._subscribe_component_trigger(bus, action_id)

            self._dbus_registered = True
            log.info("KGlobalAccel registered: %s (key_int=0x%x)", self.key_string, key_int)
            return True

        except Exception as e:
            log.warning("KGlobalAccel registration failed: %s", e)
            log.warning("Use tray menu or switch to evdev mode in config.")
            return False

    async def _subscribe_component_trigger(self, bus, action_id: list[str]) -> None:
        """Subscribe to globalShortcutPressed on /component/<name>."""
        component_path = f"/component/{action_id[0].replace('-', '_')}"
        try:
            comp_introspect = await bus.introspect("org.kde.kglobalaccel", component_path)
            comp_proxy = bus.get_proxy_object(
                "org.kde.kglobalaccel", component_path, comp_introspect
            )
            self._dbus_comp_proxy = comp_proxy  # prevent GC
            comp_iface = comp_proxy.get_interface("org.kde.kglobalaccel.Component")
            comp_iface.on_global_shortcut_pressed(self._on_component_shortcut_pressed)
            log.info("Subscribed to globalShortcutPressed at %s", component_path)
        except Exception as e:
            log.warning("Component trigger subscription failed: %s", e)

    def _on_component_shortcut_pressed(
        self, component_unique: str, shortcut_unique: str, timestamp: int
    ) -> None:
        log.info("KGlobalAccel shortcut pressed: %s/%s", component_unique, shortcut_unique)
        if shortcut_unique == "toggle-recording":
            self.recording_requested.emit()

    def _register_evdev(self) -> bool:
        """Register via evdev for press-and-hold mode."""
        try:
            import evdev
            from evdev import ecodes

            modifiers, key = parse_key_string(self.key_string)
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
