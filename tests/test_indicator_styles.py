# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Indicator-defined colours and line styles (#48)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.core import ops  # noqa: E402
from tradebot.gui.chart import ChartView  # noqa: E402
from tradebot.gui.theme import BUILTIN, ThemeManager, resolve_color  # noqa: E402
from tradebot.indicators import INDICATORS, BaseIndicator, Style  # noqa: E402
from tradebot.indicators.builtin import RSI, Bollinger  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


def test_resolve_color():
    t = BUILTIN["dark"]
    assert resolve_color(t, "up") == t.up and resolve_color(t, "accent") == t.accent
    assert resolve_color(t, "series[1]") == t.series[1]
    assert resolve_color(t, f"series[{len(t.series) + 2}]") == t.series[2]  # wraps around
    assert resolve_color(t, "#ffd166") == "#ffd166" and resolve_color(t, "rgba(1, 2, 3, 0.5)") == "rgba(1, 2, 3, 0.5)"


@pytest.fixture
def styled():
    """A plugin-style indicator with styles; unregistered afterwards so other tests don't see it."""

    class Styled(BaseIndicator):
        name = "test_styled"
        overlay = False
        outputs = ("fast", "slow", "hist")
        styles = {"fast": Style(color="#ffd166", width=3), "slow": Style(color="up", dash="dash"),
                  "hist": Style(color="down")}

        @classmethod
        def compute(cls, o):
            fast = ops.sma(o.close, 5)
            slow = ops.sma(o.close, 20)
            return {"fast": fast, "slow": slow, "hist": fast - slow}

    yield Styled
    INDICATORS._items.pop("test_styled", None)


@pytest.fixture
def chart(tmp_path):
    app = QApplication.instance() or QApplication([])
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"])
    c.resize(1000, 700)
    c.df = make_ohlc(200)
    c.show()
    app.processEvents()
    return c


def curves(plot):
    return {item.name(): item for item in plot.items if isinstance(item, pg.PlotDataItem)}


def bars(plot):
    return [item for item in plot.items if isinstance(item, pg.BarGraphItem)]


def test_styles_set_colour_width_dash_and_histogram(chart, styled):
    chart.indicators = [styled()]
    chart.redraw()
    t = BUILTIN["dark"]
    (pane,) = chart.panes
    lines = curves(pane)
    fast, slow = lines["test_styled_fast"].opts["pen"], lines["test_styled_slow"].opts["pen"]
    assert fast.color().name() == "#ffd166" and fast.widthF() == 3 and fast.style() == Qt.PenStyle.SolidLine
    assert slow.color().name() == t.up and slow.style() == Qt.PenStyle.DashLine  # theme token
    (hist,) = bars(pane)
    assert hist.opts["brush"].color().name() == t.down
    assert "color:#ffd166'" in chart._legend[pane].text and f"color:{t.up}'" in chart._legend[pane].text


def test_theme_tokens_follow_theme_changes(chart, styled):
    chart.indicators = [styled()]
    chart.redraw()
    chart.themes.apply("light")  # redraws
    (pane,) = chart.panes
    assert curves(pane)["test_styled_slow"].opts["pen"].color().name() == BUILTIN["light"].up


def test_unstyled_indicators_keep_their_palette_colours(chart):
    t = BUILTIN["dark"]
    bands, rsi = Bollinger(), RSI()
    chart.indicators = [bands, rsi]
    chart.redraw()
    rsi_color = curves(chart.panes[0])["rsi_value"].opts["pen"].color().name()
    assert rsi_color == t.series[3]  # after Bollinger's three outputs, as before
    chart.indicators = [rsi]  # Bollinger removed
    chart.redraw()
    assert curves(chart.panes[0])["rsi_value"].opts["pen"].color().name() == rsi_color  # not recoloured
    chart.indicators = [rsi, Bollinger()]  # a new indicator continues the palette
    chart.redraw()
    middle = curves(chart.price)["bbands_middle"].opts["pen"].color().name()
    assert middle not in {t.series[0], t.series[1], t.series[2], rsi_color}
