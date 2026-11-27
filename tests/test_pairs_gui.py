# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Pairs tab: the chance that the gap closes before the stop (#223), drawn and stated."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PyQt6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent, QTextDocument  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.gui.pairs import PairView  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.symbols import SymbolSearch  # noqa: E402

from .test_pairs import engine, frame_of, synthetic_pair  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


#: the user's own symbols; no network sources, so the tests stay offline and deterministic
SEARCH = SymbolSearch(lambda: ["KO", "PEP", "KDP", "BTC/USDT", "ETH/USDT"], remote=[])


@pytest.fixture
def view(app, tmp_path):
    a, b = synthetic_pair()
    v = PairView(engine(KO=frame_of(a), PEP=frame_of(b)), ThemeManager(user_dir=tmp_path), ["KO", "PEP"], SEARCH)
    v.legs = (a, b)
    yield v
    v.close()
    v.deleteLater()


def shown_text(view) -> str:
    """The numbers line as the user reads it (rendered from its HTML)."""
    doc = QTextDocument()
    doc.setHtml(view.numbers.text())
    return doc.toPlainText()


def show(view, **settings):
    for name, value in settings.items():
        getattr(view, name).setValue(value)
    view._show("KO", "PEP", 60, "rolling", *view.legs)


def test_the_chance_is_drawn_as_a_percentage(view):
    show(view)
    (curve,) = view.chance.getPlotItem().listDataItems()
    y = curve.getData()[1]
    assert np.nanmin(y) >= 0 and np.nanmax(y) <= 100 and np.isfinite(y).sum() > 100


def test_the_chance_chart_says_what_it_shows(view):
    show(view)
    (curve,) = view.chance.getPlotItem().listDataItems()
    assert curve.name() == "chance to close before the stop, %"


def test_the_chance_axis_always_shows_zero_to_a_hundred(view):
    show(view)
    low, high = view.chance.viewRange()[1]
    assert low < 0 and high > 100 and view.chance.getViewBox().state["autoRange"][1] is False


def test_the_three_charts_share_one_time_axis_position(view, app):
    show(view)
    view.resize(1200, 840)
    view.show()
    app.processEvents()  # measured as laid out, not as constructed
    widths = {plot.getAxis("left").width() for plot in (view.prices, view.zplot, view.chance)}
    assert widths == {44}


def test_the_numbers_line_states_the_half_life_now_and_the_chance(view):
    import re

    show(view, exit=0.0, horizon=3)  # the gap ends at -0.05 σ: outside an exit band of 0
    assert re.search(r"half-life now \d+\.\d candles \(last 120\) · a trade now closes to ±0 σ before ±4 σ within "
                     r"3 candles with \d+% chance", shown_text(view))


def test_a_gap_already_inside_the_exit_band_says_so(view):
    show(view, exit=0.5)  # the gap ends at -0.05 σ
    assert "the gap is inside the exit band" in shown_text(view)


def test_a_p_value_below_a_level_is_shown_whole(view):
    # "p < 0.01" was read as the start of an HTML tag, which swallowed the ✓ and the line break after it
    show(view)
    assert "p < 0.01 ✓\nhalf-life now" in shown_text(view)


def test_the_z_chart_marks_the_entries_the_exit_and_the_stop(view):
    show(view, exit=0.7, stop=3.5)
    levels = sorted(item.value() for item in view.zplot.getPlotItem().items if isinstance(item, pg.InfiniteLine))
    assert levels == [-3.5, -2.0, -0.7, 0.0, 0.7, 2.0, 3.5]


def test_an_exit_outside_the_stop_is_explained_and_nothing_is_drawn(view):
    show(view, exit=5.0, stop=4.0)
    assert "the exit must be inside the stop" in shown_text(view)
    assert view.chance.getPlotItem().listDataItems() == []


def test_too_few_candles_say_how_many_are_needed(view):
    a, b = view.legs
    view._show("KO", "PEP", 60, "rolling", a.iloc[:150], b.iloc[:150])
    assert "chance to close: unknown" in shown_text(view) and "fewer than 182 candles" in shown_text(view)


def test_a_redraw_replaces_the_chance_curve(view):
    show(view)
    show(view, horizon=5)
    assert len(view.chance.getPlotItem().listDataItems()) == 1


# -- choosing the symbols: the chart's search and symbol dialog (#220) -------------------------------------------------
def click(field):
    field.show()
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(field, press)


@pytest.mark.parametrize("field, completer", [("symbol_a", 0), ("symbol_b", 1)])
def test_both_symbol_fields_have_the_search_icon(view, field, completer):
    assert view.completers[completer].edit is getattr(view, field)
    assert view.completers[completer].dialog_action.toolTip() == "Search all symbols…"


@pytest.mark.parametrize("field, completer", [("symbol_a", 0), ("symbol_b", 1)])
def test_a_click_into_a_symbol_field_opens_the_symbol_dialog(view, field, completer):
    click(getattr(view, field))
    dialog = view.completers[completer].dialog
    assert dialog is not None and dialog.isVisible()
    dialog.close()


def test_a_symbol_picked_in_the_dialog_goes_into_its_field(view):
    click(view.symbol_b)
    view.completers[1].dialog.pick("KDP")
    assert (view.symbol_a.text(), view.symbol_b.text()) == ("KO", "KDP")


def test_a_ratio_picked_in_the_dialog_fills_both_fields(view):
    click(view.symbol_a)
    dialog = view.completers[0].dialog
    dialog.pick("BTC/USDT:ETH/USDT")  # what the dialog's ÷ button returns
    assert (view.symbol_a.text(), view.symbol_b.text()) == ("BTC/USDT", "ETH/USDT")


def test_typing_into_a_field_completes_from_the_users_symbols(view):
    completer = view.completers[1]
    view.symbol_b.setText("PE")
    view.symbol_b.textEdited.emit("PE")
    rows = [completer.model.item(r).data(completer.SYMBOL_ROLE) for r in range(completer.model.rowCount())]
    assert rows[0] == "PEP"


def test_the_app_gives_the_pairs_tab_its_shared_symbol_search(app, tmp_path):
    view = PairView(None, ThemeManager(user_dir=tmp_path), ["KO", "PEP"], SEARCH)
    assert all(c.search is SEARCH for c in view.completers)
    view.deleteLater()


def test_the_view_still_answers_which_window_it_is_in(view):
    # its "Window" spin box was stored as `window`, hiding QWidget.window(): the symbol dialog then failed with
    # "'QSpinBox' object is not callable" whenever the tab was the top-level widget
    assert view.window() is view
