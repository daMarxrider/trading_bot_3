# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Drawn UI icons (#191): one stroke weight, recoloured per theme, rendered from SVG paths with QtSvg.

The paths are the same set as in the redesign mockups (``mockups/shell.js``), 24x24 units, round caps.
"""

from __future__ import annotations

from PyQt6.QtCore import QByteArray, QObject, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

PATHS: dict[str, str] = {
    "basket": "M5 9h14l-1.5 10h-11zM9 9l3-5 3 5",
    "journal": "M6 4h11a1 1 0 0 1 1 1v15H7a1 1 0 0 1-1-1zM6 8H4M6 12H4M6 16H4M10 8h5",
    "chevl": "M15 6l-6 6 6 6",
    "rect": "M4 7h16v10H4z",
    "hline": "M3 12h18",
    "chart": "M3 3v18h18M7 15l4-5 3 3 5-7",
    "screener": "M3 5h18l-7 8v6l-4 2v-8z",
    "scanner": "M12 3a9 9 0 1 0 9 9M12 7.5a4.5 4.5 0 1 0 4.5 4.5M12 12l7-7",
    "flask": "M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3M7.5 15h9",
    "dna": "M8 3c0 5 8 5 8 9s-8 4-8 9M16 3c0 5-8 5-8 9s8 4 8 9M9 7h6M9 17h6",
    "pairs": "M4 8h13l-3-3M20 16H7l3 3",
    "wallet": "M3 7a2 2 0 0 1 2-2h13v4M3 7v11a2 2 0 0 0 2 2h15V9H5a2 2 0 0 1-2-2M16 14.5h.01",
    "news": "M5 4h12v16H7a2 2 0 0 1-2-2zM17 8h3v10a2 2 0 0 1-2 2M8 8h6M8 12h6M8 16h4",
    "depth": "M3 20V8h5v5h4v7M21 20V6h-5v6h-4v8",
    "settings": "M4 7h10M18 7h2M4 17h2M10 17h10M16 7m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0M8 17m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0",
    "info": "M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18M12 11v5M12 8h.01",
    "power": "M12 3v8M6.3 6.3a8 8 0 1 0 11.4 0",
    "plus": "M12 5v14M5 12h14",
    "close": "M6 6l12 12M18 6L6 18",
    "search": "M11 4a7 7 0 1 0 0 14a7 7 0 0 0 0-14M16 16l5 5",
    "star": "M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z",
    "cursor": "M5 3l14 8-6 2-2 6z",
    "trend": "M4 19L20 5M4 19m-1.5 0a1.5 1.5 0 1 0 3 0a1.5 1.5 0 1 0-3 0M20 5m-1.5 0a1.5 1.5 0 1 0 3 0a1.5 1.5 0 1 0-3 0",
    "fib": "M4 5h16M4 10h16M4 15h16M4 20h16",
    "text": "M5 6h14M12 6v13M9 19h6",
    "vline": "M12 3v18M9 6l3-3 3 3",
    "ruler": "M3 17L17 3l4 4L7 21zM8 12l2 2M11 9l2 2M14 6l2 2",
    "magnet": "M6 3v9a6 6 0 0 0 12 0V3h-4v9a2 2 0 0 1-4 0V3zM6 7h4M14 7h4",
    "trash": "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13",
    "layers": "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
    "code": "M8 7l-5 5 5 5M16 7l5 5-5 5M14 4l-4 16",
    "play": "M7 4l13 8-13 8z",
    "bot": "M5 9h14v10H5zM12 5v4M12 5h.01M9 13.5h.01M15 13.5h.01M9.5 16.5h5M3 13v2M21 13v2",
    "stop": "M6 6h12v12H6z",
    "download": "M12 4v11M7 11l5 5 5-5M5 20h14",
    "chev": "M6 9l6 6 6-6",
    "chevr": "M9 6l6 6-6 6",
    "folder": "M3 6a2 2 0 0 1 2-2h4l2 3h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
    "refresh": "M20 11a8 8 0 1 0-2 6M20 4v7h-7",
    "lock": "M6 11h12v9H6zM8 11V8a4 4 0 0 1 8 0v3",
    "key": "M8 11a4 4 0 1 0 0 .1M11 13l9 8M16 18l2-2",
    "db": "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
    "cpu": "M7 7h10v10H7zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4",
    "sidebar": "M3 5h18v14H3zM15 5v14",
    "moon": "M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z",
    "check": "M5 12l5 5L20 7",
    "alert": "M12 3l10 18H2zM12 10v5M12 18h.01",
    "ext": "M14 4h6v6M20 4l-9 9M18 14v5H5V6h5",
    "save": "M5 3h12l4 4v14H5zM8 3v6h8V3M8 21v-7h8v7",
    "filter": "M4 6h16M7 12h10M10 18h4",
    "grid": "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
    "clock": "M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18M12 7v5l3 2",
    "stream": "M12 12h.01M8.5 8.5a5 5 0 0 0 0 7M15.5 8.5a5 5 0 0 1 0 7M5.6 5.6a9 9 0 0 0 0 12.8M18.4 5.6a9 9 0 0 1 0 12.8",
    "bolt": "M13 3L5 14h6l-1 7 8-11h-6z",
    "copy": "M8 8h12v12H8zM4 16V4h12",
    "eye": "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0-6 0",
    "undo": "M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3",
    "merge": "M6 4v6a6 6 0 0 0 6 6h6M18 12l3 4-3 4",
    "bell": "M6 16V11a6 6 0 0 1 12 0v5l2 2H4zM10 21h4",
    "pin": "M12 17v5M8 3h8l-1 6 3 4H6l3-4z",
    "dots": "M5 12h.01M12 12h.01M19 12h.01",
    "hand": "M8 13V6a1.5 1.5 0 0 1 3 0v6M11 11V4.5a1.5 1.5 0 0 1 3 0V12M14 12V6.5a1.5 1.5 0 0 1 3 0V15a6 6 0 0 1-6 6h-1a6 6 0 0 1-5-3l-2-4a1.5 1.5 0 0 1 2.5-1.5L8 14",
}

_cache: dict[tuple[str, str, int, float], QPixmap] = {}


def _svg(name: str, color: str) -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.6" '
            f'stroke-linecap="round" stroke-linejoin="round"><path d="{PATHS[name]}"/></svg>').encode()


def pixmap(name: str, color: str = "#ffffff", size: int = 16, ratio: float = 2.0) -> QPixmap:
    """The icon ``name`` as a crisp pixmap of ``size`` logical pixels in the CSS-style ``color``."""
    key = (name, color, size, ratio)
    if key not in _cache:
        px = QPixmap(int(size * ratio), int(size * ratio))
        px.fill(Qt.GlobalColor.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        QSvgRenderer(QByteArray(_svg(name, QColor(color).name()))).render(p, QRectF(0, 0, px.width(), px.height()))
        p.end()
        px.setDevicePixelRatio(ratio)
        _cache[key] = px
    return _cache[key]


def icon(name: str, color: str = "#ffffff", size: int = 16) -> QIcon:
    return QIcon(pixmap(name, color, size))


class _Binder(QObject):
    """Keeps one button's icon in the active theme's colours; it dies with the button, which drops the connection."""

    def __init__(self, button, name: str, themes, off: str, on: str | None, size: int):
        super().__init__(button)
        self.button, self.name, self.themes, self.off, self.on, self.size = button, name, themes, off, on, size
        button.setIconSize(QSize(size, size))
        themes.changed.connect(self.refresh)
        self.refresh()

    def refresh(self, theme=None):
        t = theme or self.themes.current
        ic = QIcon()
        ic.addPixmap(pixmap(self.name, getattr(t, self.off), self.size), QIcon.Mode.Normal, QIcon.State.Off)
        if self.on:
            ic.addPixmap(pixmap(self.name, getattr(t, self.on), self.size), QIcon.Mode.Normal, QIcon.State.On)
        self.button.setIcon(ic)


def bind_icon(button, name: str, themes, off: str = "text", on: str | None = None, size: int = 16) -> None:
    """Give ``button`` the drawn icon ``name``, recoloured on every theme change.

    ``off`` / ``on`` are theme colour names for the idle and the checked state (``on`` None: the same colour)."""
    old = getattr(button, "_icon_binder", None)
    if old is not None:  # the same button changing its icon (the sidebar toggle): reuse its binder
        old.name, old.off, old.on = name, off, on
        old.refresh()
        return
    button._icon_binder = _Binder(button, name, themes, off, on, size)
