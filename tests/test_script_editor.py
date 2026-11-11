# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The in-app script editor (#115)."""

import os
import sys
import time

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtCore import Qt, QThreadPool  # noqa: E402
from PyQt6.QtGui import QTextCursor  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from tradebot import scanner, scripts  # noqa: E402
from tradebot.config import Settings  # noqa: E402
from tradebot.gui import script_editor as se  # noqa: E402
from tradebot.gui import scanner as gs  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.indicators import INDICATORS  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def themes(app, tmp_path):
    t = ThemeManager(user_dir=tmp_path / "themes")
    t.apply("dark")
    return t


@pytest.fixture
def plugins(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEBOT_PLUGIN_DIR", str(tmp_path / "plugins"))
    before = set(INDICATORS._items)
    yield tmp_path / "plugins"
    for name in set(INDICATORS._items) - before:
        INDICATORS._items.pop(name)
    for m in [m for m in sys.modules if m.startswith("tradebot_plugin_my_ind") or m.startswith("tradebot_plugin_rsi_copy")]:
        sys.modules.pop(m)


def type_text(edit, text):
    for ch in text:
        if ch == "\n":
            QTest.keyClick(edit, Qt.Key.Key_Return)
        elif ch == "\t":
            QTest.keyClick(edit, Qt.Key.Key_Tab)
        else:
            QTest.keyClick(edit, ch)


def formats(edit, line):
    block = edit.document().findBlockByNumber(line)
    return {(r.start, r.length): r.format.foreground().color().name() for r in block.layout().formats()}


def test_highlighting_keywords_strings_comments_numbers_and_triple_quotes(themes):
    edit = se.CodeEdit(themes.current)
    edit.setPlainText('def f(x):  # note\n    s = "a # b"\n    return 42 if x else None\n"""doc\nmore"""\ny = 1')
    th = themes.current
    f0 = formats(edit, 0)
    assert f0[(0, 3)] == se.qcolor(th.accent).name()  # def
    assert f0[(11, 6)] == se.qcolor(th.muted).name()  # the comment
    f1 = formats(edit, 1)
    assert f1[(8, 7)] == se.qcolor(th.up).name() and not any(start > 8 for start, _ in f1)  # # inside the string
    f2 = formats(edit, 2)
    assert f2[(11, 2)] == se.qcolor(th.series[2]).name() and f2[(4, 6)] == se.qcolor(th.accent).name()
    assert formats(edit, 4)[(0, 7)] == se.qcolor(th.up).name()  # second line of a triple-quoted string
    assert (0, 1) not in formats(edit, 5) or formats(edit, 5)[(0, 1)] != se.qcolor(th.up).name()  # back to code


def test_tab_enter_and_comment_helpers(themes, app):
    edit = se.CodeEdit(themes.current)
    edit.show()
    type_text(edit, "if x:\n")
    assert edit.toPlainText() == "if x:\n    "  # auto-indent after a colon
    type_text(edit, "y = 1\n")
    assert edit.toPlainText().endswith("    y = 1\n    ")  # keeps the indentation
    edit.setPlainText("a\n  b")
    cur = edit.textCursor()
    cur.setPosition(0)
    edit.setTextCursor(cur)
    QTest.keyClick(edit, Qt.Key.Key_Tab)
    assert edit.toPlainText().startswith("    a")  # Tab = 4 spaces
    edit.selectAll()
    edit.toggle_comment()
    assert edit.toPlainText() == "    # a\n  # b"
    edit.selectAll()
    edit.toggle_comment()
    assert edit.toPlainText() == "    a\n  b"
    edit.selectAll()
    edit._shift_lines(True)
    assert edit.toPlainText() == "        a\n      b"
    edit._shift_lines(False)
    assert edit.toPlainText() == "    a\n  b"


def test_line_numbers_follow_the_block_count(themes, app):
    edit = se.CodeEdit(themes.current)
    edit.setPlainText("\n".join("x" for _ in range(1200)))
    assert edit.viewportMargins().left() >= edit.fontMetrics().horizontalAdvance("9") * 4


def make_editor(themes, sample=None):
    return se.ScriptEditor(themes, sample)


def test_save_writes_the_file_reloads_and_marks_errors(themes, plugins, app):
    ed = make_editor(themes)
    path, source = scripts.new_script()
    reloads = []
    ed.reloaded.connect(reloads.append)
    assert ed.open_script(path, source, saved=False)
    assert ed.dirty and ed.title.text().startswith("● ")
    ed.save()
    assert path.read_text() == source and not ed.dirty and reloads == [["my_indicator"]]
    assert ed.error.text() == "Loaded my_indicator: 1 plot, 1 alert"
    good = INDICATORS.get("my_indicator")
    ed.edit.setPlainText(source.replace("return {", "retrun {"))
    ed.save()
    assert ed.error.text().startswith("SyntaxError") and ed.edit.error_line is not None
    assert INDICATORS.get("my_indicator") is good  # the last good version stays active
    assert path.read_text() != source  # ... the broken text was saved, as asked
    ed.edit.setPlainText(source.replace("ops.sma(o.close, period)", "ops.nope(o.close, period)"))
    ed.save()
    assert "dry run" in ed.error.text() and ed.edit.error_line and INDICATORS.get("my_indicator") is good
    ed.edit.setPlainText(source.replace("Close against", "Price against"))
    ed.save()
    assert ed.error.text().startswith("Loaded") and ed.edit.error_line is None
    assert INDICATORS.get("my_indicator") is not good and INDICATORS.get("my_indicator").description.startswith("Price")


def test_auto_reload_waits_for_a_pause_and_does_not_write_the_file(themes, plugins, monkeypatch, app):
    monkeypatch.setattr(se, "AUTO_RELOAD_MS", 60)
    ed = make_editor(themes)
    path, source = scripts.new_script()
    ed.open_script(path, source, saved=False)
    ed.save()
    ed.timer.setInterval(60)
    assert not ed.auto.isChecked()
    ed.edit.setPlainText(source.replace("Close against", "A"))
    time.sleep(0.1)
    QApplication.processEvents()
    assert INDICATORS.get("my_indicator").description.startswith("Close")  # off by default
    ed.auto.setChecked(True)
    ed.edit.setPlainText(source.replace("Close against", "Auto against"))
    assert ed.timer.isActive()
    end = time.time() + 3
    while INDICATORS.get("my_indicator").description.startswith("Close") and time.time() < end:
        QApplication.processEvents()
    assert INDICATORS.get("my_indicator").description.startswith("Auto")
    assert path.read_text() == source and ed.dirty and ed.title.text().startswith("●")  # not written until Ctrl+S


def test_unsaved_changes_are_asked_about_on_switching_and_closing(themes, plugins, monkeypatch, app):
    ed = make_editor(themes)
    path, source = scripts.new_script()
    ed.open_script(path, source, saved=False)
    ed.save()
    ed.edit.setPlainText(source + "\n# more")
    answers = iter([QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Discard, QMessageBox.StandardButton.Save])
    monkeypatch.setattr(QMessageBox, "exec", lambda self: next(answers))
    other = plugins / "other.py"
    assert not ed.open_script(other, "x = 1\n") and ed.path == path  # cancelled: stays
    assert ed.open_script(other, "x = 1\n") and ed.path == other and not other.exists()  # discarded
    ed.edit.setPlainText(source)
    closed = []
    ed.closed.connect(lambda: closed.append(1))
    ed.request_close()  # Save fails to load nothing meaningful but saves the file
    assert other.read_text() == source and closed == [1] and ed.path is None


# -- in the Scanner ------------------------------------------------------------------------------------
def fake_scan(symbols, indicator, params, timeframe, filters, progress=None, cancel=None):
    cls = INDICATORS.get(indicator)
    table = pd.DataFrame({"symbol": symbols, **{o: [1.0] * len(symbols) for o in cls.outputs}})
    return scanner.ScanResult(table, len(symbols), len(symbols), {}, 0.1)


@pytest.fixture
def view(themes, tmp_path, plugins, app):
    v = gs.ScannerView(None, themes, lambda: {"Main": ["AAPL", "MSFT"]}, Settings(tmp_path / "s.json"),
                       scan_fn=fake_scan, index_fn=lambda k: [])
    yield v


def wait_until(cond, timeout=5):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        QApplication.processEvents()
        QThreadPool.globalInstance().waitForDone(20)
    assert cond()


def test_picker_has_new_edit_and_copy(view, plugins):
    assert view.editor.isHidden()
    classes, hidden = gs.screenable_split()
    dlg = gs.IndicatorDialog(classes, hidden)
    actions = {r.key: r.action for r in dlg.picker.rows}
    assert actions["rsi"] == "Copy to my scripts"
    path, source = scripts.new_script()
    scripts.reload_script(path, source)
    scripts.save(path, source)
    classes, hidden = gs.screenable_split()
    dlg = gs.IndicatorDialog(classes, hidden)
    assert {r.key: r.action for r in dlg.picker.rows}["my_indicator"] == "Edit"
    assert dlg.picker.rows[0].key == "my_indicator"  # My scripts come first
    dlg.picker.action.emit("rsi")
    assert dlg.script_action == ("copy", "rsi")


def test_new_script_copy_and_edit_open_the_pane(view, plugins):
    assert view.open_script("new") and not view.editor.isHidden()
    assert view.editor.path == plugins / "my_indicator.py" and view.editor.dirty
    view.editor.save()
    assert plugins.joinpath("my_indicator.py").exists()
    assert view.open_script("copy", "rsi") and view.editor.path.name == "rsi_copy.py"
    assert "class RsiCopy(BaseIndicator)" in view.editor.edit.toPlainText()
    view.editor.save()  # the unsaved-changes prompt wasn't needed: it was saved
    assert "rsi_copy" in INDICATORS
    assert view.open_script("edit", "my_indicator") and view.editor.path.name == "my_indicator.py"


def test_saving_updates_the_scanners_indicator_columns_filters_and_stale_prompt(view, plugins):
    view.open_script("new")
    view.editor.save()
    view._set_indicator("my_indicator")
    assert view.columns == ["average"] and [c["alert"] for c in view.chips] == ["above"]
    view.start_scan()
    wait_until(lambda: view.result is not None)
    assert view.notification.isHidden()
    got = []
    view.indicator_reloaded.connect(got.append)
    src = view.editor.edit.toPlainText().replace('outputs = ("average", "above")', 'outputs = ("average", "slow", "above")') \
        .replace('return {"average": average,', 'return {"average": average, "slow": average * 2,')
    view.editor.edit.setPlainText(src)
    view.editor.save()
    assert got == [["my_indicator"]]
    assert view.cls is INDICATORS.get("my_indicator") and "slow" in view.cls.outputs
    view.set_column("slow", True)
    assert view.columns == ["average", "slow"]
    assert not view.notification.isHidden()  # the rows are from the old version: Rescan
    view.rescan_btn.click()
    wait_until(lambda: view.notification.isHidden() and not view.scanning)
    assert "slow" in view.result.table.columns


def test_a_broken_save_leaves_the_scanner_on_the_last_good_version(view, plugins):
    view.open_script("new")
    view.editor.save()
    view._set_indicator("my_indicator")
    good = view.cls
    view.editor.edit.setPlainText(view.editor.edit.toPlainText().replace("class ", "clas ", 1))
    view.editor.save()
    assert view.cls is good and INDICATORS.get("my_indicator") is good
    assert view.editor.error.text().startswith("SyntaxError")


def test_chart_redraws_a_reloaded_indicator(themes, tmp_path, plugins, app):
    from tradebot.gui.chart import ChartView
    from .conftest import make_ohlc

    path, source = scripts.new_script()
    assert scripts.reload_script(path, source).ok
    chart = ChartView(None, themes, ["TEST"])
    chart.df = make_ohlc(200)
    chart.indicators = [INDICATORS.get("my_indicator")(period=10)]
    old = chart.indicators[0]
    assert scripts.reload_script(path, source.replace("Close against", "Other")).ok
    chart.reload_indicators(["my_indicator"])
    assert chart.indicators[0] is not old and chart.indicators[0].values == {"period": 10}
    assert type(chart.indicators[0]) is INDICATORS.get("my_indicator")
