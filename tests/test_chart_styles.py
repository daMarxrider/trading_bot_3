# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Indicator colour overrides, level colours and area fills (#52)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pg = pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.style_dialog import StyleDialog  # noqa: E402
from tradebot.gui.theme import BUILTIN, ThemeManager, qcolor, resolve_color  # noqa: E402
from tradebot.indicators.builtin import RSI, Bollinger  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


def make_chart(tmp_path, themes=None):
    app = QApplication.instance() or QApplication([])
    if themes is None:
        themes = ThemeManager(user_dir=tmp_path)
        themes.apply("dark")
    c = ChartView(None, themes, ["TEST"], settings=Settings(tmp_path / "settings.json"))
    c.resize(1100, 800)
    c.show()
    c._loaded("BTC/USDT", make_ohlc(300))
    c.indicators = [Bollinger(), RSI()]
    c.redraw()
    app.processEvents()
    return c


def curve_pen(plot, name):
    return next(i for i in plot.items if isinstance(i, pg.PlotDataItem) and i.name() == name).opts["pen"]


def test_override_is_applied_and_remembered(tmp_path):
    c = make_chart(tmp_path)
    c.set_style_overrides("bbands", {"middle": {"color": "#123456", "width": 3.0, "dash": "dash"}})
    pen = curve_pen(c.price, "bbands_middle")
    assert pen.color().name() == "#123456" and pen.widthF() == 3.0
    assert curve_pen(c.price, "bbands_upper").color().name() != "#123456"
    again = Settings(tmp_path / "settings.json")  # a new session
    assert again.get("chart.styles") == {"bbands": {"middle": {"color": "#123456", "width": 3.0, "dash": "dash"}}}
    c2 = make_chart(tmp_path, c.themes)
    assert curve_pen(c2.price, "bbands_middle").color().name() == "#123456"
    c2.set_style_overrides("bbands", {}, merge=False)  # reset
    assert Settings(tmp_path / "settings.json").get("chart.styles") == {}


def test_dialog_returns_only_changes():
    QApplication.instance() or QApplication([])
    d = StyleDialog("x", [("value", "#112233", 1.6, None), ("signal", "#445566", 2.0, "dash")])
    assert d.overrides() == {}
    d._colors["value"] = "#abcdef"
    d.dashes["signal"].setCurrentIndex(0)
    assert d.overrides() == {"value": {"color": "#abcdef"}, "signal": {"dash": "solid"}}


def test_level_colours(tmp_path):
    c = make_chart(tmp_path)
    t = c.themes.current
    pens = {line.value(): line.pen for line in c.panes[0].items if isinstance(line, pg.InfiniteLine)}
    assert pens[30.0].color().name() == qcolor(t.up).name()
    assert pens[70.0].color().name() == qcolor(t.down).name()


def test_fill_drawn_and_not_fitted(tmp_path):
    c = make_chart(tmp_path)
    fills = [i for i in c.price.items if isinstance(i, pg.FillBetweenItem)]
    assert len(fills) == 1
    assert fills[0].brush().color().alpha() == qcolor("rgba(76, 201, 240, 0.08)").alpha()
    assert len(c._fit_series[c.price]) == 2 + 3  # high/low and the three band lines only


@pytest.mark.parametrize("theme", sorted(BUILTIN))
def test_every_theme_resolves_level_colours(theme):
    t = BUILTIN[theme]
    for style in (*RSI.level_styles.values(), *(f for _, _, f in Bollinger.fills)):
        assert qcolor(resolve_color(t, style.color)).isValid()
