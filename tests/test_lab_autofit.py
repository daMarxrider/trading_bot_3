# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Backtest lab y-axis auto-fit (#74): the same toggle and behaviour as in the chart."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.algorithms import build_algorithm  # noqa: E402
from tradebot.backtest import backtest  # noqa: E402
from tradebot.gui.chart import FIT_PADDING  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402

from .conftest import make_ohlc  # noqa: E402
from .test_chart_autofit import FakeWheel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themes(app, tmp_path):
    t = ThemeManager(user_dir=tmp_path)
    t.apply("dark")
    return t


@pytest.fixture
def lab(themes):
    view = BacktestView(None, themes, ["TEST"])
    view.resize(1200, 800)
    view.show()
    df = make_ohlc(400)
    split = 300  # an in-sample and an out-of-sample curve on different time ranges, like an optimizer run
    view._show((df, {"in-sample": backtest(df.iloc[:split], build_algorithm("sma_cross")),
                     "out-of-sample": backtest(df.iloc[split:], build_algorithm("sma_cross"))}))
    QApplication.processEvents()
    return view


def vb(lab):
    return lab.plot.getPlotItem().vb


def curves(lab):
    return [item.getData() for item in lab.plot.getPlotItem().listDataItems()]


def assert_fits(lab, x0, x1):
    """The y-range is exactly the padded range of what all plotted curves show between x0 and x1."""
    ys = np.concatenate([y[(x >= x0) & (x <= x1)] for x, y in curves(lab)])
    lo, hi = np.nanmin(ys), np.nanmax(ys)
    pad = FIT_PADDING * (hi - lo)
    assert vb(lab).viewRange()[1] == pytest.approx((lo - pad, hi + pad), rel=1e-6)


def test_on_by_default_in_bottom_left(lab):
    btn = lab.fit_btn
    assert btn.isCheckable() and btn.isChecked() and not btn.isHidden()
    assert btn.parent() is lab.plot and btn.x() < 20
    assert btn.geometry().bottom() > lab.plot.height() - 20


def test_a_run_fits_all_curves(lab):
    assert len(curves(lab)) == 3  # buy & hold, in-sample, out-of-sample
    assert_fits(lab, *vb(lab).viewRange()[0])


@pytest.mark.parametrize("i0, i1", [(20, 80), (280, 330), (350, 399)])
def test_zooming_time_fits_the_visible_curves(lab, i0, i1):
    x = lab._x
    lab.plot.setXRange(x[i0], x[i1], padding=0)
    assert_fits(lab, x[i0], x[i1])


def test_wheel_zooms_time_only_and_the_value_axis_is_locked(lab):
    v = vb(lab)
    center = v.boundingRect().center()
    v.wheelEvent(FakeWheel(center))
    assert_fits(lab, *v.viewRange()[0])
    fitted = v.viewRange()[1]
    v.wheelEvent(FakeWheel(center), axis=1)  # scrolling over the value axis
    assert v.viewRange()[1] == pytest.approx(fitted)
    assert lab.fit_btn.isChecked()


def test_cannot_pan_past_the_data(lab):
    x = lab._x
    span = x[50] - x[0]
    lab.plot.setXRange(x[-1] + 9 * span, x[-1] + 10 * span, padding=0)
    x0, x1 = vb(lab).viewRange()[0]
    assert x1 <= x[-1] + (x[1] - x[0]) * 1.01 and x0 < x[-1]


def test_toggle_off_unlocks_and_keeps_the_range(lab):
    v = vb(lab)
    lab.plot.setXRange(lab._x[0], lab._x[30], padding=0)
    fitted = v.viewRange()[1]
    lab.fit_btn.setChecked(False)
    assert v.state["mouseEnabled"] == [True, True]
    assert v.viewRange()[1] == pytest.approx(fitted)  # switching off doesn't jump
    v.wheelEvent(FakeWheel(v.boundingRect().center()), axis=1)
    assert v.viewRange()[1] != pytest.approx(fitted)  # now the value axis is free

    lab.fit_btn.setChecked(True)  # back on: refits and locks right away
    assert v.state["mouseEnabled"] == [True, False]
    assert_fits(lab, *v.viewRange()[0])


def test_reset_shows_everything_fitted(lab):
    lab.plot.setXRange(lab._x[100], lab._x[120], padding=0)
    lab.reset_view()  # what a double tap does
    x0, x1 = vb(lab).viewRange()[0]
    assert x0 <= lab._x[0] and x1 >= lab._x[-1]
    assert_fits(lab, x0, x1)


def test_toggle_is_remembered_across_sessions(themes, tmp_path):
    from tradebot.config import Settings

    path = tmp_path / "settings.json"
    first = BacktestView(None, themes, ["TEST"], settings=Settings(path))
    assert first.fit_btn.isChecked()
    first.fit_btn.setChecked(False)
    assert not BacktestView(None, themes, ["TEST"], settings=Settings(path)).fit_btn.isChecked()
