# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Bots page (#157): dashboard, start/stop, editor, shared limits. Fake brokers only; nothing is sent."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from tradebot.config import Settings  # noqa: E402
from tradebot.gui.bots import BotsView, parse_params  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402
from tradebot.live.manager import AccountLimits, BotManager, BotSet, BotSpec, load_bots, save_bots  # noqa: E402

from .test_bot_manager import FakeBroker  # noqa: E402
from .test_services import Scripted  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("tradebot.notify.notify", lambda *a, **k: None)
    monkeypatch.setattr("tradebot.notify.bot_event", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(BotManager, "start_loop", lambda self: None)  # the tests tick by hand, no thread
    algos = {}
    monkeypatch.setattr(BotSpec, "build_algorithm", lambda spec: algos.setdefault(spec.name, Scripted()))
    return algos


def make_view(tmp_path, bots=None, theme="graphite", settings=None):
    path = tmp_path / "bots.toml"
    if bots is not None:
        save_bots(bots, path)
    broker = FakeBroker()
    themes = ThemeManager(user_dir=tmp_path / "themes")
    themes.apply(theme)
    view = BotsView(themes, settings, path=path, interval=60_000,
                    manager_factory=lambda b: BotManager(b, broker_factory=lambda n, s: broker,
                                                         state_path=tmp_path / "state"))
    return view, broker


@pytest.fixture
def view(app, tmp_path):
    bots = BotSet([BotSpec("eth", "ETH/USDT", "sma_cross", timeframe="1d"),
                   BotSpec("btc", "BTC/USDT", "sma_cross", timeframe="1d", mode="paper"),
                   BotSpec("sol", "SOL/USDT", "sma_cross", mode="live")], AccountLimits(max_open_positions=2))
    v, broker = make_view(tmp_path, bots)
    yield v
    v.shutdown()
    v.close()
    v.deleteLater()


def test_parse_params():
    assert parse_params("fast=10, slow=30") == {"fast": 10, "slow": 30}
    assert parse_params("") == {}
    with pytest.raises(ValueError):
        parse_params("fast")
    with pytest.raises(ValueError):
        parse_params("fast=x")


def test_dashboard_lists_the_set_and_starts_and_stops(view, isolated):
    assert list(view.model.df["bot"]) == ["eth", "btc", "sol"]
    assert set(view.model.df["state"]) == {"stopped"}
    assert "≤ 2 positions" in view.kpis["limits"].value.text()
    modes = []
    view.running_changed.connect(modes.append)
    isolated["eth"] = Scripted()
    isolated["eth"].next = 1.0
    view.start_all()
    assert sorted(view.manager.running()) == ["btc", "eth"]  # the live one stays in the terminal
    assert modes[-1] == ["dry", "paper"]
    view.manager.tick()
    view.refresh()
    row = view.model.df.set_index("bot").loc["eth"]
    assert row["state"] == "running" and row["position"] == 1.0 and row["last event"].startswith("buy")
    assert view.kpis["running"].value.text() == "2 of 3" and view.kpis["positions"].value.text() == "1"
    assert len(view.events_model.df) >= 2, (view.selected(), {n: (r.error, r.events) for n, r in view._rows.items()})
    view._select("eth")
    assert view.events_title.text() == "Events · eth" and set(view.events_model.df["bot"]) == {"eth"}
    assert view.stop_btn.isEnabled() and not view.start_btn.isEnabled()
    view.stop_selected()
    assert view.manager.running() == ["btc"]
    view._select("sol")
    assert not view.start_btn.isEnabled()  # live: terminal only
    assert not view.start("sol")
    view.stop_all()
    assert view.manager.running() == [] and modes[-1] == []


def test_pnl_has_colour_and_triangle(view, isolated):
    isolated["eth"] = Scripted()
    isolated["eth"].next = 1.0
    view.start("eth")
    view.manager.tick()
    bot = view.manager._runners["eth"].bot
    bot.last_price = 90.0  # 1 ETH bought at 100
    view.refresh()
    i = list(view.model.df["bot"]).index("eth")
    col = list(view.model.df.columns).index("P&L")
    cell = view.model.data(view.model.index(i, col))
    assert cell == "▼ 10.00"
    from PyQt6.QtCore import Qt

    assert view.model.data(view.model.index(i, col), Qt.ItemDataRole.ForegroundRole).name() == \
        view.themes.current.down.lower()
    assert "▼ 10.00" in view.kpis["pnl"].value.text() and view.themes.current.down in view.kpis["pnl"].value.text()


def test_editor_adds_edits_and_deletes(view, tmp_path):
    view.new_bot()
    view.f_name.setText("ada")
    view.f_symbol.setText("ADA/USDT")
    view.f_params.setText("fast=5, slow=20")
    view.f_tf.setCurrentText("4h")
    view.f_stake.setValue(25)
    view.f_drawdown.setValue(12)
    assert view.save_bot()
    spec = load_bots(tmp_path / "bots.toml").get("ada")
    assert spec.params == {"fast": 5, "slow": 20} and spec.timeframe == "4h" and spec.stake == 25
    assert spec.risk == {"max_drawdown": 12} and spec.mode == "dry"
    assert "ada" in list(view.model.df["bot"]) and view.selected() == "ada"
    view.new_bot()
    view.f_name.setText("ada")
    view.f_symbol.setText("X")
    assert not view.save_bot()  # names are unique
    view._select("ada")
    view.f_name.setText("ada2")
    view.f_mode.setCurrentIndex(view.f_mode.findData("paper"))
    assert view.save_bot()
    names = [b.name for b in load_bots(tmp_path / "bots.toml").bots]
    assert names == ["eth", "btc", "sol", "ada2"] and load_bots(tmp_path / "bots.toml").get("ada2").mode == "paper"
    view.f_params.setText("fast")
    assert not view.save_bot()
    view._select("ada2")
    view.delete_bot()
    assert "ada2" not in [b.name for b in load_bots(tmp_path / "bots.toml").bots]


def test_editing_a_running_bot_stops_it(view, tmp_path):
    view.start("eth")
    view._select("eth")
    view.f_stake.setValue(42)
    seen = []
    view.status.connect(seen.append)
    assert view.save_bot()
    assert "eth" not in view.manager.running() and any("settings changed" in s for s in seen)


def test_shared_limits_are_saved(view, tmp_path):
    view.l_exposure.setValue(300)
    view.l_positions.setValue(0)
    view.l_daily.setValue(40)
    view.l_close.setChecked(True)
    assert view.save_limits()
    lim = load_bots(tmp_path / "bots.toml").limits
    assert lim == AccountLimits(300, 0, 40, True) and view.manager.guard.limits == lim
    assert "exposure ≤ 300" in view.kpis["limits"].value.text()


def test_broken_file_is_not_overwritten(app, tmp_path):
    (tmp_path / "bots.toml").write_text("[[bots]]\nname = 'a'\n")
    v, _ = make_view(tmp_path)
    try:
        assert "could not be read" in v.note.text()
        v.f_name.setText("b")
        v.f_symbol.setText("B/USDT")
        assert not v.save_bot()
        assert (tmp_path / "bots.toml").read_text() == "[[bots]]\nname = 'a'\n"
    finally:
        v.shutdown()
        v.close()
        v.deleteLater()


def test_empty_set_shows_the_empty_state_and_autostart(app, tmp_path):
    settings = Settings(tmp_path / "s.json")
    v, _ = make_view(tmp_path, settings=settings)
    try:
        assert v.bots_box.currentWidget() is v.bots_box.empty
        v.l_autostart.setChecked(True)
        assert settings.get("bots.autostart") is True
    finally:
        v.shutdown()
        v.close()
        v.deleteLater()


@pytest.mark.parametrize("theme", ["graphite", "paper"])
@pytest.mark.parametrize("size", [(1440, 900), (1024, 640)])
def test_fits_in_both_themes(app, tmp_path, theme, size):
    bots = BotSet([BotSpec(f"bot-{i}", "ETH/USDT", "sma_cross", params={"fast": 10, "slow": 30}) for i in range(4)])
    v, _ = make_view(tmp_path, bots, theme)
    try:
        v.resize(*size)
        v.show()
        app.processEvents()
        assert v.minimumSizeHint().width() <= size[0] - 192  # next to the sidebar
        for label in (v.note, *(k.label for k in v.kpis.values())):
            assert label.width() >= min(label.sizeHint().width(), label.width()) and label.isVisible()
        for btn in (v.start_btn, v.stop_btn, v.start_all_btn, v.stop_all_btn, v.new_btn):
            assert btn.width() >= btn.sizeHint().width()  # labels are never cut
    finally:
        v.shutdown()
        v.close()
        v.deleteLater()


def test_main_window_has_the_page_and_the_chip_names_the_bots(app, tmp_path, monkeypatch):
    from tradebot.gui import app as gui

    monkeypatch.setattr(gui.ChartView, "load", lambda self: None)
    monkeypatch.setattr(gui, "warm_up", lambda *a: None)
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "cfg" / "config.toml"))
    save_bots(BotSet([BotSpec("eth", "ETH/USDT", "sma_cross", mode="paper")]), tmp_path / "cfg" / "bots.toml")
    broker = FakeBroker()
    monkeypatch.setattr("tradebot.live.manager._default_broker", lambda name, sandbox=False: broker)
    win = gui.MainWindow(ThemeManager(user_dir=tmp_path / "themes"), "graphite", Settings(tmp_path / "s.json"))
    try:
        assert win.stack.indexOf(win.bots) >= 0 and list(win.bots.model.df["bot"]) == ["eth"]
        assert win.mode_chip.text() == "RESEARCH · NO ORDERS"
        win.bots.start("eth")
        assert win.mode_chip.text() == "PAPER BOTS · NO REAL ORDERS"
        win.kill_btn.click()
        assert win.mode_chip.text() == "KILL SWITCH ON"
        win.kill_btn.click()
        win.close()  # closing the window stops the bots
        assert win.bots.manager.running() == []
    finally:
        win.close()
        win.deleteLater()
