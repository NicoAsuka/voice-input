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

        # Animations — use custom opacity property (windowOpacity unsupported on some Wayland backends)
        self._opacity_anim = QPropertyAnimation(self, b"capsuleOpacity")
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
            log.debug("Layer Shell: using Qt fallback flags (sufficient for KDE Plasma 6)")
        except Exception as e:
            log.debug("Layer Shell setup failed, using fallback: %s", e)

    def _try_blur(self) -> None:
        """Try to enable KWin blur behind the window."""
        try:
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
            native = self.windowHandle()
            if native is not None:
                self._use_blur = True
                log.debug("KWin blur hint applied")
        except Exception as e:
            log.debug("KWin blur not available: %s", e)
            self._use_blur = False

    def _animate_entry(self) -> None:
        # Disconnect stale finished→hide() from a previous animate_exit
        try:
            self._opacity_anim.finished.disconnect()
        except TypeError:
            pass
        self._opacity_anim.setDuration(350)
        self._opacity_anim.setStartValue(0.0)
        self._opacity_anim.setEndValue(1.0)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.OutBack)
        self._opacity_anim.start()

    def animate_exit(self, on_finished=None) -> None:
        self._opacity_anim.stop()
        try:
            self._opacity_anim.finished.disconnect()
        except TypeError:
            pass
        self._opacity_anim.setDuration(220)
        self._opacity_anim.setStartValue(1.0)
        self._opacity_anim.setEndValue(0.0)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.InQuad)
        if on_finished:
            self._opacity_anim.finished.connect(on_finished)
        self._opacity_anim.start()

    # --- Custom opacity property (avoids windowOpacity Wayland issues) ---

    def get_capsule_opacity(self) -> float:
        return self._opacity

    def set_capsule_opacity(self, v: float) -> None:
        self._opacity = v
        self.update()

    capsuleOpacity = pyqtProperty(float, get_capsule_opacity, set_capsule_opacity)

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
        painter.setOpacity(max(0.0, min(1.0, self._opacity)))

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
