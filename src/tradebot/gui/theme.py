# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Theme manager: built-in Graphite / Paper (the #191 redesign) and the older Glass / Dark / Light themes, plus user themes from JSON.

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
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPainter, QRadialGradient
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
    accent_ink: str = "#ffffff"  #: text on an accent-filled button
    rail: str = ""  #: navigation rail; empty = the panel colour
    warn: str = "#e9c14c"
    live: str = "#cc2438"  #: fill of the kill-switch / live state; white text on it


PLEX = "'IBM Plex Sans', Inter, 'Segoe UI', 'SF Pro Text', 'Noto Sans', sans-serif"

BUILTIN: dict[str, Theme] = {
    # the #191 redesign: solid warm-graphite panels, one violet accent, blue-teal up / vermilion down
    "graphite": Theme(
        name="Graphite", bg="#121211", panel="#1a1a18", panel_border="#2d2c28", text="#ece9e1", muted="#a09c91",
        accent="#c4b2ff", up="#43c2d4", down="#ff8453", grid="rgba(255, 255, 255, 0.055)", input_bg="#21211e",
        selection="#2b2740", radius=8, font=PLEX, accent_ink="#17122b", rail="#0d0d0c", warn="#e9c14c", live="#cc2438",
        series=("#43c2d4", "#ff8453", "#e9c14c", "#c4b2ff", "#7ee0a8", "#f08fc0"),
    ),
    "paper": Theme(
        name="Paper", bg="#f1efe8", panel="#fbf9f3", panel_border="#dcd8cb", text="#1d1c18", muted="#5f5c52",
        accent="#5a3fd1", up="#0a7a8c", down="#bf4515", grid="rgba(0, 0, 0, 0.06)", input_bg="#f4f1e9",
        selection="#e4def8", radius=8, font=PLEX, accent_ink="#ffffff", rail="#e8e5db", warn="#916d00", live="#c8203a",
        series=("#0a7a8c", "#bf4515", "#916d00", "#5a3fd1", "#1c7f55", "#b0357f"),
    ),
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
QWidget#Sidebar {{ background: {rail_bg}; border-right: 1px solid {panel_border}; }}
QLabel#NavGroup {{ color: {muted}; font-size: 11px; font-weight: 500; padding: 12px 8px 3px 8px; }}
QFrame#StateOverlay {{ background: {panel}; border: none; }}
QLabel#EmptyTitle {{ font-size: 14px; font-weight: 600; }}
QLabel#Brand {{ font-size: 16px; font-weight: 600; }}
QFrame#Card, QGroupBox {{ background: {panel}; border: 1px solid {panel_border}; border-radius: {radius}px; }}
QGroupBox {{ margin-top: 14px; padding: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {muted}; }}
QLabel#Title {{ font-size: 18px; font-weight: 600; }}
QFrame#Strip {{ background: {panel}; border: 1px solid {panel_border}; border-radius: {radius}px; }}
QFrame#Stat {{ background: transparent; border: none; border-right: 1px solid {panel_border}; border-radius: 0; }}
QFrame#Stat[last="true"] {{ border-right: none; }}
QLabel#KpiValue {{ font-size: 16px; font-weight: 600; }}
QLabel#KpiText {{ font-size: 13px; font-weight: 500; }}
QLabel#ModeChip[state="bots"] {{ border-color: {warn}; color: {warn}; }}
QLabel#Muted, QLabel#KpiLabel {{ color: {muted}; }}
QPushButton {{ background: {input_bg}; border: 1px solid {panel_border}; border-radius: {radius_sm}px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:checked, QPushButton#Primary {{ background: {accent}; color: {accent_ink}; border-color: {accent}; }}
QPushButton#Primary {{ font-weight: 600; }}
QPushButton:disabled, QToolButton:disabled {{ color: {muted}; background: transparent; }}
QPushButton#Nav {{ background: transparent; border: none; text-align: left; padding: 7px 10px; border-radius: {radius_sm}px; color: {muted}; }}
QPushButton#Nav:hover {{ background: {nav_hover}; color: {text}; }}
QPushButton#Nav:checked {{ background: {selection}; color: {text}; font-weight: 500; border: none; }}
QPushButton#Kill {{ background: transparent; border: 1px solid {panel_border}; border-radius: 4px; padding: 1px 8px; color: {muted}; min-height: 0; }}
QPushButton#Kill:hover {{ border-color: {live}; color: {text}; }}
QPushButton#Kill:checked {{ background: {live}; border-color: {live}; color: #ffffff; font-weight: 600; }}
QLabel#ModeChip {{ border: 1px solid {panel_border}; border-radius: 4px; padding: 1px 8px; font-weight: 600; font-size: 11px; background: {input_bg}; }}
QLabel#ModeChip[state="kill"] {{ background: {live}; border-color: {live}; color: #ffffff; }}
QFrame#KillBanner {{ background: {live}; }}
QFrame#KillBanner QLabel {{ color: #ffffff; font-weight: 500; }}
QToolButton {{ background: {input_bg}; border: 1px solid {panel_border}; border-radius: {radius_sm}px; padding: 5px 9px; }}
QToolButton:checked {{ background: {accent}; color: {accent_ink}; }}
QToolButton::menu-indicator {{ image: none; }}
QToolButton#ChartCorner {{ padding: 0; min-height: 0; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {muted}; border: none; border-bottom: 2px solid transparent; padding: 7px 12px; margin-right: 2px; }}
QTabBar::tab:hover {{ color: {text}; }}
QTabBar::tab:selected {{ color: {text}; font-weight: 500; border-bottom: 2px solid {accent}; }}
QFrame#Chip {{ border: 1px solid {panel_border}; border-radius: {radius_sm}px; }}
QFrame#Chip QToolButton {{ background: transparent; border: none; padding: 3px 6px; }}
QFrame#ChartSidebar {{ background: {panel}; border: 1px solid {panel_border}; border-radius: {radius}px; }}
QFrame#ChartSidebar QToolButton#SidebarToggle, QFrame#ChartSidebar QToolButton#SidebarTab {{
    background: transparent; border: none; border-radius: {radius_sm}px; padding: 0; min-height: 0; font-size: 24px; }}
QFrame#ChartSidebar QToolButton#SidebarToggle:hover, QFrame#ChartSidebar QToolButton#SidebarTab:hover {{ background: {selection}; }}
QFrame#ChartSidebar QToolButton#SidebarTab:checked {{ background: {selection}; color: {text}; }}
QLabel#SidebarTitle {{ font-weight: 600; font-size: 14px; }}
QListWidget#SidebarNews {{ background: transparent; border: none; }}
QListWidget#SidebarNews::item:hover {{ background: {selection}; border-radius: {radius_sm}px; }}
QListWidget#Watchlist {{ background: transparent; border: none; }}
QListWidget#Watchlist::item:hover, QListWidget#Watchlist::item:selected {{ background: {selection}; border-radius: {radius_sm}px; }}
QToolButton#WatchlistRemove {{ background: transparent; border: none; color: {muted}; }}
QToolButton#WatchlistRemove:hover {{ color: {down}; }}
QLineEdit, QComboBox, QAbstractSpinBox, QPlainTextEdit {{
    background: {input_bg}; border: 1px solid {panel_border}; border-radius: {radius_sm}px; padding: 5px 8px;
    selection-background-color: {selection};
}}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus, QPlainTextEdit:focus {{ border-color: {accent}; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none; background: transparent; }}
QDateEdit::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none; background: transparent; }}
QDateEdit::down-arrow {{ image: url({spin_down}); width: 10px; height: 10px; }}
QComboBox::down-arrow {{ image: url({spin_down}); width: 10px; height: 10px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; border: 1.5px solid {border_strong}; background: {input_bg}; }}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; image: url({check_img}); }}
QRadioButton::indicator:checked {{ background: {accent}; border-color: {accent}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ border-color: {panel_border}; }}
QAbstractSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 18px; border: none; background: transparent; }}
QAbstractSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 18px; border: none; background: transparent; }}
QAbstractSpinBox::up-arrow {{ image: url({spin_up}); width: 10px; height: 10px; }}
QAbstractSpinBox::down-arrow {{ image: url({spin_down}); width: 10px; height: 10px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {selection}; }}
QCalendarWidget QWidget {{ background: {menu_bg}; color: {text}; alternate-background-color: {menu_bg}; }}
QCalendarWidget QWidget#qt_calendar_navigationbar {{ border-bottom: 1px solid {panel_border}; }}
QCalendarWidget QToolButton {{ background: transparent; border: none; color: {text}; font-weight: 600; padding: 4px 8px; }}
QCalendarWidget QToolButton:hover {{ background: {selection}; }}
QCalendarWidget QToolButton#qt_calendar_prevmonth, QCalendarWidget QToolButton#qt_calendar_nextmonth {{ font-size: 18px; }}
QCalendarWidget QAbstractItemView {{ selection-background-color: {accent}; selection-color: {accent_ink}; outline: 0; }}
QCalendarWidget QAbstractItemView:disabled {{ color: {muted}; }}
QComboBox QAbstractItemView, QMenu, QAbstractItemView#SymbolPopup {{ background: {menu_bg}; border: 1px solid {panel_border}; selection-background-color: {selection}; }}
QMenu::item {{ padding: 5px 22px; }}
QMenu QPushButton#LoadoutName {{ background: transparent; border: none; border-radius: 0; text-align: left; padding: 5px 22px; }}
QMenu QPushButton#LoadoutName:hover {{ background: {selection}; }}
QMenu QToolButton#LoadoutDelete {{ background: transparent; border: none; color: {muted}; padding: 2px 8px; }}
QMenu QToolButton#LoadoutDelete:hover {{ color: {down}; }}
QMenu::item:selected {{ background: {selection}; }}
QTableView, QListWidget {{ background: {panel}; border: 1px solid {panel_border}; border-radius: {radius}px;
    gridline-color: {grid}; selection-background-color: {selection}; selection-color: {text}; alternate-background-color: {alt_row}; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: transparent; color: {muted}; border: none; border-bottom: 1px solid {panel_border}; padding: 6px; }}
QTableCornerButton::section {{ background: transparent; border: none; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover, QSplitter::handle:pressed {{ background: {selection}; border-radius: 3px; }}
QScrollArea#Pane {{ background: transparent; border: none; }}
QScrollArea#Pane > QWidget#qt_scrollarea_viewport {{ background: transparent; }}
QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {panel_border}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QProgressBar {{ background: {input_bg}; border: none; border-radius: 4px; height: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}
QStatusBar {{ color: {muted}; background: {status_bg}; border-top: 1px solid {panel_border}; }}
QStatusBar::item {{ border: none; }}
QProgressBar {{ text-align: center; }}
QToolTip {{ background: {menu_bg}; color: {text}; border: 1px solid {panel_border}; }}
QDialog#SymbolDialog, QDialog#StudyDialog {{ background: {menu_bg}; border: 1px solid {border_strong}; }}
QFrame#SymbolSearchBar {{ border: none; border-bottom: 1px solid {panel_border}; }}
QLineEdit#SymbolQuery {{ background: transparent; border: none; padding: 0; font-size: 15px; font-weight: 500; }}
QFrame#RatioToken {{ background: {selection}; border-radius: {radius_sm}px; }}
QFrame#RatioToken QLabel {{ background: transparent; font-weight: 600; }}
QToolButton#SymbolIconButton {{ background: transparent; border: none; border-radius: {radius_sm}px; padding: 4px; }}
QToolButton#SymbolIconButton:hover {{ background: {nav_hover}; }}
QLabel#RatioNote {{ background: {selection}; color: {text}; padding: 7px 14px; }}
QListWidget#SymbolRail {{ background: {rail_bg}; border: none; border-right: 1px solid {panel_border}; border-radius: 0; padding: 8px; outline: 0; }}
QListWidget#SymbolRail::item {{ border-radius: {radius_sm}px; color: {muted}; padding-left: 10px; }}
QListWidget#SymbolRail::item:hover {{ background: {nav_hover}; }}
QListWidget#SymbolRail::item:selected {{ background: {selection}; }}
QTableView#SymbolTable {{ background: transparent; border: none; border-radius: 0; outline: 0; }}
QTableView#SymbolTable::item {{ padding: 0 6px; border: none; }}
QTableView#SymbolTable::item:selected {{ background: {selection}; color: {text}; }}
QTableView#SymbolTable QHeaderView::section {{ padding: 6px 6px; }}
QLabel#SymbolEmpty {{ color: {muted}; padding: 40px; }}
QFrame#SymbolFooter {{ border: none; border-top: 1px solid {panel_border}; }}
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


def _luma(color: str) -> float:
    c = QColor(color)
    return (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()) / 255


def _mix(base: str, other: str, amount: float) -> str:
    """``base`` moved ``amount`` of the way towards ``other``; a translucent (rgba) base is returned as it is."""
    if base.startswith("rgba") or other.startswith("rgba"):
        return base
    a, b = QColor(base), QColor(other)
    return QColor(*(round(x + (y - x) * amount) for x, y in ((a.red(), b.red()), (a.green(), b.green()), (a.blue(), b.blue())))).name()


def _opaque(color: str, fallback: str) -> str:
    return fallback if color.startswith("rgba") else color


def stylesheet(t: Theme, touch: bool = False) -> str:
    return (QSS + (TOUCH_QSS if touch else "")).format(
        **asdict(t),
        root_bg="transparent" if t.glass else t.bg,
        radius_sm=max(4, t.radius // 2),
        border_strong=_mix(t.panel_border, t.text, 0.3),
        check_img=(ICON_DIR / ("check-light.svg" if _luma(t.accent_ink) > 0.5 else "check-dark.svg")).as_posix(),
        spin_up=(ICON_DIR / "spin-up.svg").as_posix(),
        spin_down=(ICON_DIR / "spin-down.svg").as_posix(),
        rail_bg=t.rail or (t.bg if t.glass else t.panel),
        nav_hover=t.selection if t.glass else t.input_bg,
        status_bg=t.bg if t.glass else (t.rail or t.panel),  # the status bar sits outside the glass backdrop
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


FONT_DIR = Path(__file__).with_name("assets") / "fonts"
ICON_DIR = Path(__file__).with_name("assets") / "icons"
_fonts_loaded = False


def load_fonts() -> None:
    """Register the bundled IBM Plex fonts with Qt, once (#191). A missing file just means the fallback fonts."""
    global _fonts_loaded
    if _fonts_loaded:
        return
    _fonts_loaded = True
    for path in sorted(FONT_DIR.glob("*.ttf")):
        if QFontDatabase.addApplicationFont(str(path)) < 0:
            log.warning("could not load font %s", path)


class ThemeManager(QObject):
    changed = pyqtSignal(object)

    def __init__(self, user_dir: Path | None = None, touch: bool = False):
        super().__init__()
        self.touch = touch
        self.themes: dict[str, Theme] = dict(BUILTIN)
        self.themes.update(self._load_user_themes(user_dir or Path("~/.config/tradebot/themes").expanduser()))
        self.current: Theme = self.themes["graphite"]

    @staticmethod
    def _load_user_themes(directory: Path) -> dict[str, Theme]:
        out = {}
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
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
        load_fonts()
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

    # with real compositor blur behind the window (#7) the gradient is only a tint on top of it
    BLUR_BASE_ALPHA = 150
    BLUR_BLOB_ALPHA = 70

    def __init__(self, manager: ThemeManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.blur_active = False
        self.setObjectName("Root")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        manager.changed.connect(lambda _: self.update())

    def paintEvent(self, event):
        t = self.manager.current
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        base = QColor(t.bg)
        blurred = self.blur_active and t.glass
        if blurred:
            base.setAlpha(self.BLUR_BASE_ALPHA)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.fillRect(self.rect(), base)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if t.glass:
            w, h = self.width(), self.height()
            for fx, fy, fr, color in self.BLOBS:
                grad = QRadialGradient(QPointF(fx * w, fy * h), fr * max(w, h))
                c = QColor(color)
                c.setAlpha(self.BLUR_BLOB_ALPHA if blurred else 110)
                grad.setColorAt(0.0, c)
                c.setAlpha(0)
                grad.setColorAt(1.0, c)
                p.fillRect(self.rect(), grad)
        p.end()
