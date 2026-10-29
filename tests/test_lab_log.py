# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The backtest lab's logarithmic value scale (#76), like the chart's (#60)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.algorithms import build_algorithm  # noqa: E402
from tradebot.backtest import backtest  # noqa: E402
from tradebot.config import Settings  # noqa: E402
from tradebot.gui.chart import FIT_PADDING  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402

from .test_chart_log import growth  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themes(app, tmp_path):
    t = ThemeManager(user_dir=tmp_path)
    t.apply("dark")
    return t


@pytest.fixture
def lab(themes, tmp_path):
    view = BacktestView(None, themes, ["TEST"], settings=Settings(tmp_path / "settings.json"))
    view.resize(1200, 800)
    view.show()
    df = growth()
    view._show((df, {"in-sample": backtest(df.iloc[:300], build_algorithm("sma_cross")),
                     "out-of-sample": backtest(df.iloc[300:], build_algorithm("sma_cross"))}))
    QApplication.processEvents()
    return view


def vb(lab):
    return lab.plot.getPlotItem().vb


def curves(lab):
    return [item.getData() for item in lab.plot.getPlotItem().listDataItems()]


def test_button_next_to_autofit_off_by_default_and_remembered(themes, tmp_path):
    path = tmp_path / "s.json"
    first = BacktestView(None, themes, ["TEST"], settings=Settings(path))
    first.resize(1000, 700)
    first.show()
    QApplication.processEvents()
    btn, fit = first.log_btn, first.fit_btn
    assert btn.isCheckable() and not btn.isChecked() and btn.parent() is first.plot
    assert btn.y() == fit.y() and btn.x() > fit.geometry().right()  # right of ↕, same row
    assert btn.size() == fit.size()
    btn.setChecked(True)
    assert BacktestView(None, themes, ["TEST"], settings=Settings(path)).log_btn.isChecked()


def test_curves_are_drawn_as_log10_and_fitted(lab):
    linear = curves(lab)
    lab.log_btn.setChecked(True)
    assert lab.value_axis.log
    for (_, y_lin), (_, y_log) in zip(linear, curves(lab)):
        assert y_log == pytest.approx(np.log10(y_lin))
    ys = np.concatenate([y for _, y in curves(lab)])
    lo, hi = ys.min(), ys.max()
    pad = FIT_PADDING * (hi - lo)
    assert vb(lab).viewRange()[1] == pytest.approx((lo - pad, hi + pad), rel=1e-6)


def test_zoomed_log_view_fits_the_visible_part(lab):
    lab.log_btn.setChecked(True)
    x = lab._x
    lab.plot.setXRange(x[0], x[50], padding=0)
    ys = np.concatenate([y[(cx >= x[0]) & (cx <= x[50])] for cx, y in curves(lab)])
    pad = FIT_PADDING * (ys.max() - ys.min())
    assert vb(lab).viewRange()[1] == pytest.approx((ys.min() - pad, ys.max() + pad), rel=1e-6)


def test_toggling_keeps_the_time_range(lab):
    x = lab._x
    lab.plot.setXRange(x[100], x[200], padding=0)
    before = tuple(vb(lab).viewRange()[0])
    lab.log_btn.setChecked(True)
    assert tuple(vb(lab).viewRange()[0]) == pytest.approx(before)
    lab.log_btn.setChecked(False)
    assert tuple(vb(lab).viewRange()[0]) == pytest.approx(before)
    assert vb(lab).viewRange()[1][1] > 100  # linear values again


def test_ticks_show_real_values(lab):
    lab.log_btn.setChecked(True)
    assert lab.value_axis.tickStrings([np.log10(2000.0)], 1, 1) == ["2,000"]


def test_values_at_or_below_zero_fall_back_to_linear(lab):
    messages = []
    lab.status.connect(messages.append)
    name, x, y, pen = lab._curves[0]
    lab._curves[0] = (name, x, np.where(np.arange(len(y)) == 5, 0.0, y), pen)
    lab.log_btn.setChecked(True)
    assert not lab._log and not lab.value_axis.log
    assert any("linear" in m for m in messages)
