# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Theme manager: built-in Glass / Dark / Light themes plus user themes from JSON.

A theme is a palette of tokens; one QSS template turns it into a stylesheet,
and the same tokens style the pyqtgraph charts, so every widget follows the
active theme. User themes live in ``~/.config/tradebot/themes/*.json``::

    {"name": "Midnight", "base": "dark", "accent": "#ff6ad5", "bg": "#0b0b1a"}

Any token omitted falls back to the ``base`` theme.

The *glass* theme renders translucent, rounded "frosted" panels above a soft,
blurred-light gradient backdrop drawn by :class:`GlassBackdrop`. It is painted
by the app itself, so it looks the same on every OS and needs no compositor.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PyQt6.QtCore import QObject, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QRadialGradient
from PyQt6.QtWidgets import QApplication, QWidget

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str  #: window background (behind panels)
    panel: str  #: card/panel fill, may be translucent rgba
    panel_border: str
    text: str
    muted: str
    accent: str
    up: str  #: rising candles / positive numbers
    down: str  #: falling candles / negative numbers
    grid: str
    input_bg: str
    selection: str
    radius: int = 10
    glass: bool = False
    font: str = "Inter, 'Segoe UI', 'SF Pro Text', 'Noto Sans', sans-serif"
    series: tuple[str, ...] = ("#4cc9f0", "#f72585", "#ffd166", "#06d6a0", "#b388ff", "#ff9f1c")


BUILTIN: dict[str, Theme] = {
    "glass": Theme(
        name="Glass", bg="#0d1022", panel="rgba(255, 255, 255, 0.07)", panel_border="rgba(255, 255, 255, 0.16)",
        text="#eef1ff", muted="#9aa3c7", accent="#7c9cff", up="#2ee6a6", down="#ff5c8a",
        grid="rgba(255, 255, 255, 0.07)", input_bg="rgba(255, 255, 255, 0.08)", selection="rgba(124, 156, 255, 0.35)",
        radius=16, glass=True,
    ),
    "dark": Theme(
        name="Modern Dark", bg="#0f1115", panel="#171a21", panel_border="#262b36", text="#e6e8ee", muted="#8b93a7",
        accent="#3d8bfd", up="#26a69a", down="#ef5350", grid="#232733", input_bg="#1d212a", selection="#2b4a7d",
        radius=8,
    ),
    "light": Theme(
        name="Light", bg="#f3f5f9", panel="#ffffff", panel_border="#dde2ec", text="#1b2130", muted="#667089",
        accent="#2563eb", up="#0f9d76", down="#dc3545", grid="#e7ebf2", input_bg="#f7f9fc", selection="#cfe0ff",
        radius=8, series=("#2563eb", "#db2777", "#d97706", "#059669", "#7c3aed", "#ea580c"),
    ),
}

QSS = """
* {{ font-family: {font}; font-size: 13px; color: {text}; }}
QMainWindow, #Root {{ background: {root_bg}; }}
QDialog {{ background: {bg}; }}
QWidget#Sidebar {{ background: {panel}; border-right: 1px solid {panel_border}; }}
QFrame#Card, QGroupBox {{ background: {panel}; border: 1px solid {panel_border}; border-radius: {radius}px; }}
QGroupBox {{ margin-top: 14px; padding: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {muted}; }}
QLabel#Title {{ font-size: 20px; font-weight: 600; }}
QLabel#KpiValue {{ font-size: 22px; font-weight: 600; }}
QLabel#Muted, QLabel#KpiLabel {{ color: {muted}; }}
QPushButton {{ background: {input_bg}; border: 1px solid {panel_border}; border-radius: {radius_sm}px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:checked, QPushButton#Primary {{ background: {accent}; color: #ffffff; border-color: {accent}; }}
QPushButton#Nav {{ background: transparent; border: none; text-align: left; padding: 10px 16px; border-radius: {radius_sm}px; }}
QPushButton#Nav:hover {{ background: {selection}; }}
QPushButton#Nav:checked {{ background: {selection}; color: {text}; font-weight: 600; }}
QToolButton {{ background: {input_bg}; border: 1px solid {panel_border}; border-radius: {radius_sm}px; padding: 5px 9px; }}
QToolButton:checked {{ background: {accent}; color: #ffffff; }}
QToolButton::menu-indicator {{ image: none; }}
QLineEdit, QComboBox, QAbstractSpinBox, QPlainTextEdit {{
    background: {input_bg}; border: 1px solid {panel_border}; border-radius: {radius_sm}px; padding: 5px 8px;
    selection-background-color: {selection};
}}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus, QPlainTextEdit:focus {{ border-color: {accent}; }}
QCalendarWidget QWidget {{ background: {menu_bg}; color: {text}; alternate-background-color: {menu_bg}; }}
QCalendarWidget QWidget#qt_calendar_navigationbar {{ border-bottom: 1px solid {panel_border}; }}
QCalendarWidget QToolButton {{ background: transparent; border: none; color: {text}; font-weight: 600; padding: 4px 8px; }}
QCalendarWidget QToolButton:hover {{ background: {selection}; }}
QCalendarWidget QToolButton#qt_calendar_prevmonth, QCalendarWidget QToolButton#qt_calendar_nextmonth {{ font-size: 18px; }}
QCalendarWidget QAbstractItemView {{ selection-background-color: {accent}; selection-color: #ffffff; outline: 0; }}
QCalendarWidget QAbstractItemView:disabled {{ color: {muted}; }}
QComboBox QAbstractItemView, QMenu, QAbstractItemView#SymbolPopup {{ background: {menu_bg}; border: 1px solid {panel_border}; selection-background-color: {selection}; }}
QMenu::item {{ padding: 5px 22px; }}
QMenu::item:selected {{ background: {selection}; }}
QTableView, QListWidget {{ background: {panel}; border: 1px solid {panel_border}; border-radius: {radius}px;
    gridline-color: {grid}; selection-background-color: {selection}; alternate-background-color: {alt_row}; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: transparent; color: {muted}; border: none; border-bottom: 1px solid {panel_border}; padding: 6px; }}
QTableCornerButton::section {{ background: transparent; border: none; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {panel_border}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QProgressBar {{ background: {input_bg}; border: none; border-radius: 4px; height: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}
QStatusBar {{ color: {muted}; background: {status_bg}; border-top: 1px solid {panel_border}; }}
QToolTip {{ background: {menu_bg}; color: {text}; border: 1px solid {panel_border}; }}
"""


#: Appended in touch mode: finger-sized hit targets (~40 px) and grabbable scrollbars.
TOUCH_QSS = """
QPushButton, QToolButton {{ min-height: 30px; padding: 8px 16px; }}
QPushButton#Nav {{ min-height: 34px; padding: 12px 16px; }}
QLineEdit, QComboBox, QAbstractSpinBox {{ min-height: 30px; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 28px; }}
QComboBox QAbstractItemView::item, QMenu::item, QAbstractItemView#SymbolPopup::item {{ min-height: 36px; padding: 6px 24px; }}
QCheckBox::indicator {{ width: 24px; height: 24px; }}
QTableView {{ font-size: 14px; }}
QHeaderView::section {{ padding: 10px 6px; }}
QScrollBar:vertical, QScrollBar:horizontal {{ width: 18px; height: 18px; }}
QScrollBar::handle {{ border-radius: 9px; min-height: 48px; min-width: 48px; }}
"""


def _opaque(color: str, fallback: str) -> str:
    return fallback if color.startswith("rgba") else color


def stylesheet(t: Theme, touch: bool = False) -> str:
    return (QSS + (TOUCH_QSS if touch else "")).format(
        **asdict(t),
        root_bg="transparent" if t.glass else t.bg,
        radius_sm=max(4, t.radius // 2),
        status_bg=t.bg if t.glass else t.panel,  # the status bar sits outside the glass backdrop
        menu_bg=_opaque(t.panel, "#1a1e36"),
        alt_row="rgba(255,255,255,0.025)" if t.glass else t.input_bg,
    )


THEME_COLORS = ("up", "down", "accent", "muted", "text", "grid", "panel_border", "selection")


def resolve_color(t: Theme, spec: str) -> str:
    """A style colour for theme ``t``: theme tokens (``"up"``, ``"series[2]"``, …) or a literal CSS colour."""
    if spec.startswith("series[") and spec.endswith("]"):
        return t.series[int(spec[7:-1]) % len(t.series)]
    return getattr(t, spec) if spec in THEME_COLORS else spec


def qcolor(css: str) -> QColor:
    """Parse ``#rrggbb`` or ``rgba(r, g, b, a)`` (a in 0..1)."""
    if css.startswith("rgba"):
        r, g, b, a = (float(v) for v in css[css.index("(") + 1 : -1].split(","))
        return QColor(int(r), int(g), int(b), int(a * 255))
    return QColor(css)


class ThemeManager(QObject):
    changed = pyqtSignal(object)

    def __init__(self, user_dir: Path | None = None, touch: bool = False):
        super().__init__()
        self.touch = touch
        self.themes: dict[str, Theme] = dict(BUILTIN)
        self.themes.update(self._load_user_themes(user_dir or Path("~/.config/tradebot/themes").expanduser()))
        self.current: Theme = self.themes["glass"]

    @staticmethod
    def _load_user_themes(directory: Path) -> dict[str, Theme]:
        out = {}
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            try:
                data = json.loads(path.read_text())
                base = BUILTIN.get(data.pop("base", "dark"), BUILTIN["dark"])
                if "series" in data:
                    data["series"] = tuple(data["series"])
                out[path.stem] = replace(base, **data)
            except Exception:
                log.exception("invalid theme file %s", path)
        return out

    def apply(self, key: str) -> Theme:
        theme = self.themes[key]
        self.current = theme
        app = QApplication.instance()
        app.setStyleSheet(stylesheet(theme, self.touch))
        font = QFont()
        font.setFamilies([f.strip().strip("'") for f in theme.font.split(",")])
        app.setFont(font)
        self.changed.emit(theme)
        return theme


class GlassBackdrop(QWidget):
    """Window background: base colour plus large, soft colour blobs.

    Painted with wide radial gradients, which read as heavily blurred light
    behind the translucent panels (the glassmorphism look) at negligible cost.
    For non-glass themes it simply fills the theme background.
    """

    BLOBS = ((0.12, 0.18, 0.55, "#5b5bf7"), (0.85, 0.12, 0.45, "#c13cff"),
             (0.70, 0.90, 0.60, "#00b3c7"), (0.20, 0.95, 0.40, "#ff4f8b"))

    def __init__(self, manager: ThemeManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setObjectName("Root")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        manager.changed.connect(lambda _: self.update())

    def paintEvent(self, event):
        t = self.manager.current
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(t.bg))
        if t.glass:
            w, h = self.width(), self.height()
            for fx, fy, fr, color in self.BLOBS:
                grad = QRadialGradient(QPointF(fx * w, fy * h), fr * max(w, h))
                c = QColor(color)
                c.setAlpha(110)
                grad.setColorAt(0.0, c)
                c.setAlpha(0)
                grad.setColorAt(1.0, c)
                p.fillRect(self.rect(), grad)
        p.end()
