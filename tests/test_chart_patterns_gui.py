"""Pattern markers and support/resistance bands on the chart (#164)."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import ChartView, ZonesItem, zone_segments  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators import INDICATORS  # noqa: E402


def test_zone_segments_split_on_changes_and_gaps():
    x = np.arange(8, dtype=float)
    y = np.array([np.nan, 10, 10, 12, 12, np.nan, 12, 12])
    half = y * 0.1
    segs = zone_segments(x, y, half, 1.0)
    assert [(s[0], s[1], s[4]) for s in segs] == [(0.5, 2.5, 10), (2.5, 4.5, 12), (5.5, 7.5, 12)]
    assert segs[0][2] == pytest.approx(9) and segs[0][3] == pytest.approx(11)
    assert zone_segments(x, np.full(8, np.nan), half, 1.0) == []


def series(n=320):
    rng = np.random.default_rng(3)
    t = np.arange(n)
    close = 100 + 12 * np.sin(t / 22) + 6 * np.sin(t / 7) + np.cumsum(rng.normal(0, .5, n)) * .5
    spread = np.abs(rng.normal(0, .7, n)) + .3
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close,
                         "volume": rng.lognormal(10, .5, n)},
                        index=pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC"))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def chart(app, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    c = ChartView(None, themes, ["TEST"], Settings(tmp_path / "s.json"))
    c.resize(1000, 700)
    c.df = series()
    c.show()
    yield c
    c.close()
    c.deleteLater()


def draw(chart, *names):
    chart.indicators = [INDICATORS.get(n)() for n in names]
    chart.redraw(autorange=True)
    QApplication.processEvents()


def test_levels_are_drawn_as_bands_not_lines(chart):
    draw(chart, "support_resistance", "volume_levels")
    zones = [i for i in chart.price.items if isinstance(i, ZonesItem)]
    assert len(zones) == 5 and all(z.segments for z in zones)
    assert not [i for i in chart.price.items if isinstance(i, pg.PlotDataItem) and i.name()]  # no line curves


def test_pattern_labels_pair_colour_with_a_triangle_and_explain_themselves(chart):
    draw(chart, "candles", "chart_patterns")
    texts = [i for i in chart.price.items if isinstance(i, pg.TextItem)]
    assert texts
    theme = chart.themes.current
    for t in texts:
        text = t.toPlainText()
        assert text[0] in "▲▼◆" and t.toolTip()
        colour = t.textItem.defaultTextColor().name()
        assert colour == {"▲": theme.up, "▼": theme.down, "◆": theme.muted}[text[0]].lower()


def test_pattern_labels_on_one_side_keep_their_distance(chart):
    draw(chart, "candles")
    x = chart._x
    for sign in "▲▼":
        xs = sorted(t.pos().x() for t in chart.price.items
                    if isinstance(t, pg.TextItem) and t.toPlainText()[0] == sign)
        gaps = np.diff(np.searchsorted(x, xs))
        assert len(gaps) == 0 or gaps.min() >= chart.LABEL_GAP
