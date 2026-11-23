# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The Graphite redesign (#191): themes, drawn icons, bundled fonts, nav rail, status-bar mode chip and kill banner."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
from PyQt6.QtGui import QFontDatabase  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui import app as gui  # noqa: E402
from tradebot.gui.icons import PATHS, icon, pixmap  # noqa: E402
from tradebot.gui.theme import BUILTIN, FONT_DIR, ThemeManager, load_fonts  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _contrast(a: str, b: str) -> float:
    def lum(h: str) -> float:
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [v / 12.92 if v <= .03928 else ((v + .055) / 1.055) ** 2.4 for v in c]
        return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + .05) / (lo + .05)


@pytest.mark.parametrize("key", ["graphite", "paper"])
def test_new_themes_are_readable(key):
    t = BUILTIN[key]
    assert _contrast(t.text, t.panel) >= 7 and _contrast(t.muted, t.panel) >= 4.5
    assert _contrast(t.accent_ink, t.accent) >= 4.5 and _contrast("#ffffff", t.live) >= 4.5
    assert t.up != t.down and not t.glass


def test_graphite_is_the_default_theme(tmp_path):
    assert ThemeManager(user_dir=tmp_path).current.name == "Graphite"
    assert next(iter(BUILTIN)) == "graphite"


def test_icons_render_in_a_given_colour(qapp):
    assert all(not icon(n, "#c4b2ff").isNull() for n in ("chart", "settings", "power"))
    px = pixmap("chart", "#ff0000").toImage()
    assert any(px.pixelColor(x, y).alpha() > 0 for x in range(px.width()) for y in range(px.height()))


def test_bundled_fonts_register(qapp):
    load_fonts()
    assert sorted(p.name for p in FONT_DIR.glob("*.ttf")) == [
        "IBMPlexMono-Bold.ttf", "IBMPlexMono-Regular.ttf", "IBMPlexSans-Medium.ttf", "IBMPlexSans-Regular.ttf",
        "IBMPlexSans-SemiBold.ttf"]
    assert "IBM Plex Sans" in QFontDatabase.families() and (FONT_DIR / "OFL.txt").is_file()
    assert "IBM Plex Mono" in QFontDatabase.families()
    assert PATHS  # the icon set ships with the package


def test_rail_groups_and_kill_banner(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(gui.ChartView, "load", lambda self: None)
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    win = gui.MainWindow(ThemeManager(user_dir=tmp_path / "themes"), "graphite", Settings(tmp_path / "s.json"))
    try:
        heads = [w.text() for w in win.findChildren(QLabel) if w.objectName() == "NavGroup"]
        assert heads == ["Markets", "Strategies", "Account"]
        navs = [b for b in win.findChildren(QPushButton) if b.objectName() == "Nav"]
        assert all(not b.icon().isNull() for b in navs)
        assert win.mode_chip.text() == "RESEARCH · NO ORDERS" and win.kill_banner.isHidden()
        win.kill_btn.click()
        assert not win.kill_banner.isHidden() and win.mode_chip.text() == "KILL SWITCH ON"
        assert win.kill_btn.text() == "Kill switch ON"
        win.kill_btn.click()
        assert win.kill_banner.isHidden() and win.mode_chip.text() == "RESEARCH · NO ORDERS"
    finally:
        win.close()
        win.deleteLater()


def test_signed_columns_get_an_up_or_down_triangle(qapp, tmp_path):
    import pandas as pd
    from PyQt6.QtCore import Qt

    from tradebot.gui.common import DataFrameModel
    from tradebot.gui.views import _signed_color

    themes = ThemeManager(user_dir=tmp_path)
    df = pd.DataFrame({"sym": ["A", "B", "C"], "change_1d": [0.021, -0.013, 0.0], "price": [10.0, 20.0, 30.0]})
    model = DataFrameModel(df, {"change_1d": lambda v: f"{v:+.1%}"}, colors=_signed_color(themes, {"change_1d"}))
    text = [model.data(model.index(r, 1), Qt.ItemDataRole.DisplayRole) for r in range(3)]
    assert text == ["▲ +2.1%", "▼ -1.3%", "+0.0%"]  # zero is neither
    assert model.data(model.index(0, 2), Qt.ItemDataRole.DisplayRole) == "10"  # unmarked columns stay plain
