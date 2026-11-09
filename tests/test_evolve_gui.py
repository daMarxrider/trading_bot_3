# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The Evolve page (#83): settings, a run, the per-part curves, stopping, saving as a plugin."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.algorithms import ALGORITHMS  # noqa: E402
from tradebot.gui import evolve as page  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


class Engine:
    def get_ohlc(self, symbol, tf, start=None, end=None, bars=None):
        return make_ohlc(900, seed=sum(map(ord, symbol)))


def sync(work, done, fail):
    """run_async without a thread, errors reported like the real one."""
    try:
        result = work()
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
    else:
        done(result)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app, tmp_path, monkeypatch):
    themes = ThemeManager(user_dir=tmp_path / "themes")
    themes.apply("dark")
    v = page.EvolveView(Engine(), themes, ["AAPL"])
    v.show()
    v.population.setValue(16)
    v.generations.setValue(3)
    v.min_trades.setValue(0)
    v.seed.setValue(1)
    monkeypatch.setattr(page, "run_async", sync)
    monkeypatch.setenv("TRADEBOT_PLUGIN_DIR", str(tmp_path / "plugins"))
    QApplication.processEvents()
    return v


def test_settings_from_the_form(view):
    kw = view.settings_kwargs()
    assert kw["validation"] == 0.2 and kw["test"] == 0.2 and kw["segments"] == 3 and kw["state"]
    assert kw["indicators"] is None and kw["seed"] == 1 and kw["fee"] == 0.001  # all indicators: no restriction
    assert "elliott" not in view.indicators.boxes and "rsi" in view.indicators.boxes  # only usable ones offered
    view.indicators.set_selected(["rsi", "ema"])
    view.seed.setValue(0)
    kw = view.settings_kwargs()
    assert sorted(kw["indicators"]) == ["ema", "rsi"] and kw["seed"] is None  # random


def test_from_scratch_box(view):
    assert not view.from_scratch.isChecked() and view.indicators_card.isEnabled()
    view.from_scratch.setChecked(True)
    assert not view.indicators_card.isEnabled()
    assert (view.max_depth.value(), view.max_size.value()) == (12, 100)
    assert view.permutations.value() == 5 and view.folds.value() == 3  # the safeguards on
    kw = view.settings_kwargs()
    assert kw["from_scratch"] and kw["indicators"] is None and kw["islands"] == 4
    view.max_depth.setValue(9)  # a value the user changed stays when switching back
    view.from_scratch.setChecked(False)
    assert view.indicators_card.isEnabled() and (view.max_depth.value(), view.max_size.value()) == (9, 40)
    assert view.permutations.value() == 0 and view.folds.value() == 0 and view.islands.value() == 1
    assert not view.settings_kwargs()["from_scratch"]


def test_a_run_from_scratch(view):
    view.from_scratch.setChecked(True)
    view.permutations.setValue(1)
    view.folds.setValue(2)
    steps = []
    view.generation.connect(lambda step, best: steps.append(step))
    view.run()
    res = view.result
    assert res is not None and res.from_scratch and view.progress.maximum() == 12 and max(steps) == 11
    assert "evolved building blocks" in view.info.text()
    text = view.warnings.text()
    assert "Deflated Sharpe ratio" in text and "Walk-forward" in text


def test_one_symbol(view):
    statuses = []
    view.status.connect(statuses.append)
    assert not view.save_btn.isEnabled() and not view.chart_btn.isEnabled()
    view.run()
    res = view.result
    assert res is not None and "entry =" in view.rules.text() and "seed 1" in view.info.text()
    assert view.save_btn.isEnabled() and view.chart_btn.isEnabled() and view.run_btn.isEnabled()
    assert statuses[-1].startswith("done")
    assert len(view.per_symbol.df) == 1 and "sharpe test" in view.per_symbol.df.columns
    assert len(view.hall.df) == len(res.hall_of_fame) and "→" in set(view.hall.df[""])
    curves = view.curves("AAPL")
    assert {"train", "validation", "test", "buy & hold", "monthly investing"} <= set(curves)
    assert curves["validation"].iloc[0] == pytest.approx(curves["train"].iloc[-1])  # continues the train curve
    shaded = [i for i in view.equity.getPlotItem().items if isinstance(i, pg.LinearRegionItem)]
    assert len(shaded) == 2  # validation and test
    assert len(view.history.getPlotItem().listDataItems()) == 2  # training and validation per generation


def test_several_symbols_and_switching(view):
    view.symbol.setText("AAPL, MSFT")
    view.run()
    df = view.per_symbol.df
    assert list(df["symbol"]) == ["AAPL", "MSFT", "mean"] and "beats on" in df["vs buy & hold"].iloc[-1]
    view.show_symbol("MSFT")
    assert view._shown == "MSFT" and view.equity.getPlotItem().titleLabel.text == "MSFT"


def test_permutations_and_progress(view):
    view.permutations.setValue(1)
    steps = []
    view.generation.connect(lambda step, best: steps.append(step))
    view.run()
    assert view.progress.maximum() == 6 and max(steps) == 5  # 3 generations, twice
    assert "Shuffled prices" in view.warnings.text() and view.warnings.isVisibleTo(view)


def test_stop(view):
    statuses = []
    view.status.connect(statuses.append)
    view.generation.connect(lambda *a: view.stop())  # stop after the first generation
    view.run()
    assert view.result is None and statuses[-1] == "evolving stopped" and view.run_btn.isEnabled()


def test_save_as_plugin_and_show_on_chart(view, tmp_path):
    view.run()
    saved, shown = [], []
    view.saved.connect(saved.append)
    view.show_on_chart.connect(lambda s, a: shown.append((s, a)))
    path = view.save_plugin("evolved_gui_test")
    assert path == tmp_path / "plugins" / "evolved_gui_test.py" and path.exists()
    assert saved == ["evolved_gui_test"] and "evolved_gui_test" in ALGORITHMS  # usable without a restart
    for bad in ("sma_cross", "Bad-Name", "evolved_gui_test"):  # taken, invalid, exists
        with pytest.raises(ValueError):
            view.save_plugin(bad)
    view.save_plugin("evolved_gui_test", overwrite=True)
    view.chart_btn.click()
    symbol, algo = shown[0]
    assert symbol == "AAPL" and algo.run(make_ohlc(300)).isin([0.0, 1.0]).all()
    assert "evolved_preview" not in ALGORITHMS  # the preview isn't added to the lists


def test_main_window_has_the_page_and_lists_saved_strategies(app, tmp_path, monkeypatch):
    from tradebot.config import Settings
    from tradebot.gui import app as gui

    monkeypatch.setattr(gui.ChartView, "load", lambda self: None)  # no network
    monkeypatch.setattr(page, "run_async", sync)
    monkeypatch.setenv("TRADEBOT_PLUGIN_DIR", str(tmp_path / "plugins"))
    window = gui.MainWindow(ThemeManager(user_dir=tmp_path / "themes"), "dark", Settings(tmp_path / "s.json"))
    window.evolve.engine = Engine()
    window.evolve.population.setValue(12)
    window.evolve.generations.setValue(2)
    window.evolve.min_trades.setValue(0)
    window.evolve.run()
    window.evolve.save_plugin("evolved_window_test")
    assert window.lab.algo.findText("evolved_window_test") >= 0
    assert window.chart.algo_box.findText("evolved_window_test") >= 0
    window.close()


def test_best_result_and_fast_presets_fill_the_form(app):
    view = page.EvolveView(Engine(), ThemeManager(), ["AAA"])
    view.use_best_result()
    assert view.population.value() > 300 and view.permutations.value() >= 19 and view.folds.value() >= 3
    kw = view.settings_kwargs()
    assert kw["hall_of_fame"] == 100 and kw["tournament"] == 5
    view.use_fast()
    assert (view.population.value(), view.generations.value(), view.permutations.value()) == (300, 40, 0)
    assert "hall_of_fame" not in view.settings_kwargs()


def test_auto_configure_fills_the_form_and_reports(app, monkeypatch):
    from tradebot.evolve import auto_configure as real

    monkeypatch.setattr(page, "run_async", sync)
    monkeypatch.setattr("tradebot.evolve.auto_configure",
                        lambda df, **kw: real(df, candidates=3, rounds=2, population=20, generations=4, **kw))
    view = page.EvolveView(Engine(), ThemeManager(), ["AAA"])
    view.bars.setValue(900)
    view.auto_btn.click()
    assert view.auto_btn.isEnabled() and view.run_btn.isEnabled() and not view.stop_btn.isEnabled()
    assert "kept out" in view.info.text() and view.population.value() == 20
    assert view.permutations.value() >= 19
