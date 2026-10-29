# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The chart's logarithmic price scale (#60)."""

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
from tradebot.gui.chart import FIT_PADDING, ChartView, PriceAxis, price_label  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators.builtin import RSI, Bollinger  # noqa: E402


def growth(n=400, start=100.0, end=100_000.0) -> pd.DataFrame:
    """A thousandfold rise, like Bitcoin's history."""
    close = np.geomspace(start, end, n) * (1 + 0.02 * np.sin(np.arange(n)))
    open_ = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.01,
                         "low": np.minimum(open_, close) * 0.99, "close": close, "volume": 1.0}, index=idx)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themes(app, tmp_path):
    t = ThemeManager(user_dir=tmp_path)
    t.apply("dark")
    return t


@pytest.fixture
def chart(themes, tmp_path):
    c = ChartView(None, themes, ["TEST"], Settings(tmp_path / "settings.json"))
    c.resize(1000, 700)
    c.df = growth()
    c.show()
    c.redraw(autorange=True)
    QApplication.processEvents()
    return c


def y_range(chart):
    return tuple(chart.price.vb.viewRange()[1])


def test_button_next_to_autofit_off_by_default_and_remembered(themes, tmp_path):
    path = tmp_path / "s.json"
    first = ChartView(None, themes, ["TEST"], Settings(path))
    first.resize(800, 600)
    first.show()
    QApplication.processEvents()
    btn, fit = first.log_btn, first.fit_btn
    assert btn.isCheckable() and not btn.isChecked() and btn.parent() is first.canvas
    assert btn.y() == fit.y() and btn.x() > fit.geometry().right()  # right of ↕, same row
    assert btn.size() == fit.size()
    btn.setChecked(True)
    assert ChartView(None, themes, ["TEST"], Settings(path)).log_btn.isChecked()


def test_log_scale_fits_the_candles_in_log_space(chart):
    chart.log_btn.setChecked(True)
    lo, hi = np.log10(chart.df.low.min()), np.log10(chart.df.high.max())
    pad = FIT_PADDING * (hi - lo)
    assert y_range(chart) == pytest.approx((lo - pad, hi + pad), rel=1e-6)
    x = chart._x
    chart.price.setXRange(x[0], x[50], padding=0)  # zoom into the early, cheap part: still fits
    v = chart.df.iloc[:51]
    lo, hi = np.log10(v.low.min()), np.log10(v.high.max())
    pad = FIT_PADDING * (hi - lo)
    assert y_range(chart) == pytest.approx((lo - pad, hi + pad), rel=1e-6)


def test_toggling_keeps_the_time_range(chart):
    x = chart._x
    chart.price.setXRange(x[100], x[200], padding=0)
    before = tuple(chart.price.vb.viewRange()[0])
    chart.log_btn.setChecked(True)
    assert tuple(chart.price.vb.viewRange()[0]) == pytest.approx(before)
    chart.log_btn.setChecked(False)
    assert tuple(chart.price.vb.viewRange()[0]) == pytest.approx(before)
    assert y_range(chart)[1] > 1000  # linear prices again


def test_overlays_and_markers_on_the_log_scale_panes_stay_linear(chart):
    chart.indicators = [Bollinger(), RSI()]
    chart.log_btn.setChecked(True)
    bands = Bollinger()(chart.df).iloc[:, 0].to_numpy()
    curves = [i for i in chart.price.items if isinstance(i, pg.PlotDataItem)]
    ys = curves[0].getData()[1]
    assert ys == pytest.approx(np.log10(bands[~np.isnan(bands)]))
    (pane,) = chart.panes
    lo, hi = pane.vb.viewRange()[1]
    assert hi > 10  # RSI's own 0..100 values, not their log10


def test_the_readouts_show_prices(chart):
    chart.log_btn.setChecked(True)
    chart.hline.setVisible(True)  # as when the cursor is over the price pane
    chart.hline.setPos(np.log10(250.0))
    assert chart.hline.label.toPlainText() == "250"
    assert chart.price_at(np.log10(1234.5)) == pytest.approx(1234.5)
    chart.indicators = [Bollinger()]
    chart.redraw()
    ((_, values),) = chart._rows[chart.price]
    bands = Bollinger()(chart.df).to_numpy()
    assert np.nanmax(values[0][0]) == pytest.approx(np.nanmax(bands[:, 0]))  # the value line reads prices


def test_drawings_stay_at_their_prices(chart):
    chart._add_level(500)
    level = chart.drawings[-1]
    rect = pg.RectROI([chart._x[10], 200], [86400 * 5, 300])
    trend = pg.LineSegmentROI([[chart._x[0], 100], [chart._x[100], 1000]])
    for item, tool in ((rect, "rect"), (trend, "trend")):
        chart._add_drawing(item, tool)
    chart.log_btn.setChecked(True)
    assert level.value() == pytest.approx(np.log10(500))
    assert level.label.toPlainText() == "500"
    assert rect.pos().y() == pytest.approx(np.log10(200))
    assert rect.pos().y() + rect.size().y() == pytest.approx(np.log10(500))
    ends = [trend.mapToParent(h.pos()).y() for h in trend.getHandles()]
    assert ends == pytest.approx([2, 3])
    chart._add_level(2000)  # a level added on the log scale is at its price too
    assert chart.drawings[-1].value() == pytest.approx(np.log10(2000))
    chart.log_btn.setChecked(False)
    assert level.value() == pytest.approx(500)
    assert rect.size().y() == pytest.approx(300)
    assert [trend.mapToParent(h.pos()).y() for h in trend.getHandles()] == pytest.approx([100, 1000])
    assert chart.drawings[-1].value() == pytest.approx(2000)


def test_prices_at_or_below_zero_fall_back_to_linear(chart):
    messages = []
    chart.status.connect(messages.append)
    df = growth()
    df.iloc[5, df.columns.get_loc("low")] = 0.0
    chart.df = df
    chart.log_btn.setChecked(True)
    assert not chart._log and not chart.price_axis.log
    assert any("linear" in m for m in messages)


# -- price axis ticks --------------------------------------------------------------------------------
@pytest.fixture
def axis(themes):
    a = PriceAxis("right")
    a.log = True
    return a


def prices(levels):
    return [round(10 ** v, 6) for _, values in levels for v in values]


def test_ticks_are_round_prices(axis):
    ticks = prices(axis.tickValues(np.log10(900), np.log10(110_000), 600))
    assert ticks == [1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
    assert axis.tickStrings([np.log10(v) for v in ticks[:3]], 1, 1) == ["1,000", "2,000", "5,000"]


def test_ticks_thin_out_over_many_decades(axis):
    ticks = prices(axis.tickValues(np.log10(0.01), np.log10(1e6), 200))  # room for 5 labels
    assert 2 <= len(ticks) <= 5 and all(np.log10(t) == pytest.approx(round(np.log10(t))) for t in ticks)


def test_narrow_range_uses_linear_steps(axis):
    ticks = prices(axis.tickValues(np.log10(180), np.log10(260), 400))
    assert len(ticks) >= 3 and all(180 <= t <= 260 for t in ticks)
    assert all(t == round(t, 0) for t in ticks)  # round steps like 200, 220, …


def test_price_label():
    assert [price_label(p) for p in (100_000, 1000, 250, 0.05, 1.5)] == ["100,000", "1,000", "250", "0.05", "1.5"]
