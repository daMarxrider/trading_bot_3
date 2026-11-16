# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Position sizing in the backtest lab (#147)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.algorithms.sizing import Sized  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.gui.views import BacktestView  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lab(app, tmp_path):
    themes = ThemeManager(user_dir=tmp_path)
    themes.apply("dark")
    view = BacktestView(None, themes, ["TEST"])
    view.wf_folds.setValue(0)
    view.show()
    view.algo.setCurrentText("sma_cross")
    QApplication.processEvents()
    yield view
    view.close()  # a shown view freed by Python's refcount segfaults in sip (#143)
    view.deleteLater()
    QApplication.processEvents()


def test_full_size_is_the_default_and_adds_no_parameters(lab):
    assert lab.sizing.currentData() == "none"
    assert not isinstance(lab._algorithm(), Sized)
    assert not any(k.startswith("size.") for k in lab._algorithm().param_space())


def test_choosing_a_model_adds_its_parameters_and_wraps_the_algorithm(lab):
    lab.sizing.setCurrentIndex(lab.sizing.findData("vol"))
    QApplication.processEvents()
    algo = lab._algorithm()
    assert isinstance(algo, Sized) and algo.name == "sma_cross"
    assert {"size.target_vol", "size.window"} <= set(lab.params.values())
    lab.sizing.setCurrentIndex(lab.sizing.findData("none"))
    QApplication.processEvents()
    assert not any(k.startswith("size.") for k in lab.params.values())
