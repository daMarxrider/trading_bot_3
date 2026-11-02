# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Evolve page y-axis auto-fit (#96): the same toggle and behaviour as in the chart and the Backtest lab."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui import evolve as page  # noqa: E402
from tradebot.gui.chart import FIT_PADDING  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402

from .test_chart_autofit import FakeWheel  # noqa: E402
from .test_evolve_gui import Engine, sync  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themes(app, tmp_path):
    t = ThemeManager(user_dir=tmp_path / "themes")
    t.apply("dark")
    return t


@pytest.fixture
def view(themes, monkeypatch):
    v = page.EvolveView(Engine(), themes, ["AAPL"])
    v.resize(1400, 900)
    v.show()
    v.population.setValue(16)
    v.generations.setValue(4)
    v.min_trades.setValue(0)
    v.seed.setValue(1)
    monkeypatch.setattr(page, "run_async", sync)
    v.run()
    QApplication.processEvents()
    return v


def vb(plot):
    return plot.getPlotItem().vb


def curves(plot):
    return [item.getData() for item in plot.getPlotItem().listDataItems()]


def assert_fits(plot, x0, x1):
    """The y-range is exactly the padded range of what all plotted curves show between x0 and x1
    (on the log scale, getData() already returns log10 values)."""
    ys = np.concatenate([y[(x >= x0) & (x <= x1)] for x, y in curves(plot)]).astype(float)
    lo, hi = np.nanmin(ys), np.nanmax(ys)
    pad = FIT_PADDING * ((hi - lo) or abs(hi) * 0.01 or 1.0)
    assert vb(plot).viewRange()[1] == pytest.approx((lo - pad, hi + pad), rel=1e-6)


def test_common_x_aligns_curves_on_different_times():
    t = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    x, (a, b) = page.on_common_x([pd.Series([1.0, 2.0], t[:2]), pd.Series([3.0, 4.0, 5.0], t[1:])])
    assert len(x) == 4
    np.testing.assert_array_equal(a, [1, 2, np.nan, np.nan])
    np.testing.assert_array_equal(b, [np.nan, 3, 4, 5])


def test_both_plots_have_the_toggle_on_by_default_in_bottom_left(view):
    for plot, fit in ((view.equity, view.equity_fit), (view.history, view.history_fit)):
        btn = fit.button
        assert btn.isCheckable() and btn.isChecked() and not btn.isHidden()
        assert btn.parent() is plot and btn.x() < 20
        assert btn.geometry().bottom() > plot.height() - 20


def test_a_run_fits_all_curves(view):
    assert len(curves(view.equity)) >= 4  # train, validation, test and the baselines
    assert_fits(view.equity, *vb(view.equity).viewRange()[0])
    assert_fits(view.history, *vb(view.history).viewRange()[0])


@pytest.mark.parametrize("i0, i1", [(20, 120), (500, 650), (750, 899)])
def test_zooming_time_fits_the_visible_curves(view, i0, i1):
    x = view.equity_fit.x
    view.equity.setXRange(x[i0], x[i1], padding=0)
    assert_fits(view.equity, x[i0], x[i1])


def test_history_zoom_fits_the_visible_generations(view):
    view.history.setXRange(2, 4, padding=0)
    assert_fits(view.history, 2, 4)


def test_log_scale_fits_in_log_coordinates(view):
    view.log_btn.setChecked(True)
    assert view.equity.getPlotItem().ctrl.logYCheck.isChecked()
    x = view.equity_fit.x
    view.equity.setXRange(x[100], x[400], padding=0)
    assert_fits(view.equity, x[100], x[400])


def test_wheel_zooms_time_only_and_the_value_axis_is_locked(view):
    v = vb(view.equity)
    center = v.boundingRect().center()
    v.wheelEvent(FakeWheel(center))
    assert_fits(view.equity, *v.viewRange()[0])
    fitted = v.viewRange()[1]
    v.wheelEvent(FakeWheel(center), axis=1)  # scrolling over the value axis
    assert v.viewRange()[1] == pytest.approx(fitted)


def test_cannot_pan_past_the_data(view):
    x = view.equity_fit.x
    span = x[50] - x[0]
    view.equity.setXRange(x[-1] + 9 * span, x[-1] + 10 * span, padding=0)
    x0, x1 = vb(view.equity).viewRange()[0]
    assert x1 <= x[-1] + (x[1] - x[0]) * 1.01 and x0 < x[-1]


def test_toggle_off_unlocks_and_keeps_the_range(view):
    v, fit = vb(view.equity), view.equity_fit
    view.equity.setXRange(fit.x[0], fit.x[30], padding=0)
    fitted = v.viewRange()[1]
    fit.button.setChecked(False)
    assert v.state["mouseEnabled"] == [True, True]
    assert v.viewRange()[1] == pytest.approx(fitted)  # switching off doesn't jump
    v.wheelEvent(FakeWheel(v.boundingRect().center()), axis=1)
    assert v.viewRange()[1] != pytest.approx(fitted)  # now the value axis is free

    fit.button.setChecked(True)  # back on: refits and locks right away
    assert v.state["mouseEnabled"] == [True, False]
    assert_fits(view.equity, *v.viewRange()[0])


def test_reset_shows_everything_fitted(view):
    fit = view.equity_fit
    view.equity.setXRange(fit.x[100], fit.x[120], padding=0)
    fit.reset()  # what a double click or double tap does
    x0, x1 = vb(view.equity).viewRange()[0]
    assert x0 <= fit.x[0] and x1 >= fit.x[-1]
    assert_fits(view.equity, x0, x1)


def test_toggle_is_remembered_across_sessions(themes, tmp_path):
    from tradebot.config import Settings

    path = tmp_path / "settings.json"
    first = page.EvolveView(Engine(), themes, ["AAPL"], settings=Settings(path))
    assert first.equity_fit.button.isChecked() and first.history_fit.button.isChecked()
    first.equity_fit.button.setChecked(False)
    again = page.EvolveView(Engine(), themes, ["AAPL"], settings=Settings(path))
    assert not again.equity_fit.button.isChecked() and again.history_fit.button.isChecked()
