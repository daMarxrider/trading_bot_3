# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Several bots at once (#157): the bot set file, shared limits, the manager, the state files. Fake brokers only."""

import json
import os

import pandas as pd
import pytest

from tradebot.core.models import Balance, Order, OrderStatus, OrderType, Side, Ticker
from tradebot.live import manager as mgr
from tradebot.live.manager import AccountLimits, BotManager, BotSet, BotSpec, load_bots, save_bots

from .test_services import Scripted


@pytest.fixture(autouse=True)
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "cfg" / "config.toml"))
    monkeypatch.delenv("TRADEBOT_BOTS", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("tradebot.notify.notify", lambda *a, **k: None)
    monkeypatch.setattr("tradebot.notify.bot_event", lambda *a, **k: None)


class FakeBroker:
    """A broker that never talks to anything: daily candles at ``price``, orders fill at ``price``."""

    simulated = True
    can_trade = True
    name = "fake"

    def __init__(self, price=100.0):
        self.price, self.orders, self.held = price, [], {}

    def get_klines(self, symbol, timeframe, limit):
        now = pd.Timestamp.now(tz="UTC").floor("1D")
        idx = pd.date_range(end=now - pd.Timedelta(days=1), periods=20, freq="1D")
        p = self.price[symbol] if isinstance(self.price, dict) else self.price
        return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=idx)

    def get_ticker(self, symbol):
        return Ticker(symbol, float(self.get_klines(symbol, "1d", 1)["close"].iloc[-1]))

    def place_order(self, symbol, side, amount, type=OrderType.MARKET, price=None):
        fill = float(self.get_klines(symbol, "1d", 1)["close"].iloc[-1])
        base = symbol.split("/")[0]
        self.held[base] = self.held.get(base, 0) + (amount if side == Side.BUY else -amount)
        self.orders.append((symbol, side, amount))
        return Order(str(len(self.orders)), symbol, side, type, amount, fill, OrderStatus.CLOSED)

    def get_balance(self):
        return {k: Balance(k, v, 0) for k, v in self.held.items()}


@pytest.fixture
def algos(monkeypatch):
    """Each bot gets a Scripted algorithm (target set by the test) instead of a registered one."""
    made = {}

    def build(spec):
        made[spec.name] = made.get(spec.name) or Scripted()
        return made[spec.name]

    monkeypatch.setattr(BotSpec, "build_algorithm", build)
    return made


def manager(bots, broker, tmp_path, **kw):
    return BotManager(bots, broker_factory=lambda name, sandbox: broker, state_path=tmp_path / "state", **kw)


def test_spec_validation():
    with pytest.raises(ValueError, match="mode"):
        BotSpec("a", "ETH/USDT", "sma_cross", mode="real")
    with pytest.raises(ValueError, match="stake"):
        BotSpec("a", "ETH/USDT", "sma_cross", stake=0)
    with pytest.raises(ValueError, match="timeframe"):
        BotSpec("a", "ETH/USDT", "sma_cross", timeframe="7x")
    with pytest.raises(ValueError, match="risk limit"):
        BotSpec("a", "ETH/USDT", "sma_cross", risk={"max_loss": 1})
    with pytest.raises(ValueError, match="name"):
        BotSpec(" ", "ETH/USDT", "sma_cross")


def test_bot_set_round_trips_through_the_file(tmp_path):
    s = BotSet([BotSpec("eth trend", "ETH/USDT", "sma_cross", stake=50, params={"fast": 10, "size.risk": 0.5},
                        risk={"max_drawdown": 10}, sizing="vol_target"),
                BotSpec("btc", "BTC/USDT", "rsi", mode="paper", enabled=False, members=["a", "b"])],
               AccountLimits(max_total_exposure=500, max_open_positions=2, close_on_halt=True))
    path = save_bots(s)
    assert path == tmp_path / "cfg" / "bots.toml"
    back = load_bots()
    assert back.bots == s.bots and back.limits == s.limits
    assert "mode" not in path.read_text().split("[[bots]]")[1]  # defaults are left out


def test_load_rejects_duplicates_and_unknown_keys(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text('[[bots]]\nname="a"\nsymbol="X"\nalgorithm="rsi"\n[[bots]]\nname="a"\nsymbol="Y"\nalgorithm="rsi"\n')
    with pytest.raises(ValueError, match="more than one"):
        load_bots(p)
    p.write_text('[[bots]]\nname="a"\nsymbol="X"\nalgorithm="rsi"\nstak=1\n')
    with pytest.raises(ValueError, match="unknown key"):
        load_bots(p)
    assert load_bots(tmp_path / "missing.toml").bots == []


def test_put_and_remove():
    s = BotSet([BotSpec("a", "X", "rsi"), BotSpec("b", "Y", "rsi")])
    s.put(BotSpec("c", "Z", "rsi"), old_name="a")
    assert [b.name for b in s.bots] == ["c", "b"]
    with pytest.raises(ValueError, match="already"):
        s.put(BotSpec("b", "Z", "rsi"), old_name="c")
    s.remove("b")
    assert [b.name for b in s.bots] == ["c"]


def test_runs_several_bots_and_writes_state(tmp_path, algos):
    broker = FakeBroker({"ETH/USDT": 100.0, "BTC/USDT": 200.0})
    s = BotSet([BotSpec("eth", "ETH/USDT", "x", timeframe="1d", stake=100),
                BotSpec("btc", "BTC/USDT", "x", timeframe="1d", stake=100, mode="paper")])
    m = manager(s, broker, tmp_path)
    assert m.start_all() == {}
    algos["eth"].next = algos["btc"].next = 1.0
    events = m.tick()
    assert {n: e.action for n, e in events} == {"eth": "buy", "btc": "buy"}
    assert broker.orders == [("BTC/USDT", Side.BUY, 0.5)]  # the dry run sends nothing
    assert m.tick() == []  # not due again before the next candle
    broker.price["ETH/USDT"] = 110.0
    rows = {r.name: r for r in m.snapshot()}
    assert rows["eth"].state == "running" and rows["eth"].position == 1.0
    state = json.loads((tmp_path / "state" / "eth.json").read_text())
    assert state["pid"] == os.getpid() and state["position"] == 1.0 and state["events"][-1]["action"] == "buy"
    summary = m.summary()
    assert summary["running"] == 2 and summary["positions"] == 2 and summary["exposure"] == pytest.approx(200)
    m.stop("eth")
    assert m.running() == ["btc"]
    assert {r.name: r.state for r in m.snapshot()}["eth"] == "stopped"


def test_shared_exposure_and_position_limits(tmp_path, algos):
    broker = FakeBroker()
    s = BotSet([BotSpec(n, f"{n}/USDT", "x", timeframe="1d", stake=100) for n in ("A", "B", "C")],
               AccountLimits(max_open_positions=2))
    m = manager(s, broker, tmp_path)
    m.start_all()
    for a in ("A", "B", "C"):
        algos[a].next = 1.0
    ev = dict(m.tick())
    assert [ev[n].action for n in "ABC"] == ["buy", "buy", "hold"] and "2 bots already hold" in ev["C"].note
    s2 = BotSet([BotSpec(n, f"{n}/USDT", "x", timeframe="1d", stake=100) for n in ("A", "B")],
                AccountLimits(max_total_exposure=150))
    m2 = manager(s2, broker, tmp_path / "2")
    m2.start_all()
    ev = dict(m2.tick())
    assert ev["A"].action == "buy" and ev["B"].action == "hold" and "exposure" in ev["B"].note


def test_shared_daily_loss_halts_every_bot(tmp_path, algos):
    broker = FakeBroker({"A/USDT": 100.0, "B/USDT": 100.0})
    s = BotSet([BotSpec(n, f"{n}/USDT", "x", timeframe="1d", stake=100) for n in "AB"],
               AccountLimits(max_daily_loss=15, close_on_halt=True))
    m = manager(s, broker, tmp_path)
    m.start_all()
    algos["A"].next = algos["B"].next = 1.0
    m.tick()
    broker.price["A/USDT"] = broker.price["B/USDT"] = 92.0  # -8 each, -16 together
    for r in m._runners.values():
        r.next_due = 0
    ev = dict(m.tick())
    assert "halted" in ev["A"].note or "halted" in ev["B"].note
    assert m.guard.halted and "shared limit 15" in m.guard.halted
    assert {r.state for r in m.snapshot()} == {"halted"}
    for r in m._runners.values():  # the bot that stepped first sells at its next check
        r.next_due = 0
    assert {e.action for _, e in m.tick()} <= {"sell", "hold"}
    assert all(m._runners[n].bot.holding == 0 for n in "AB")  # close_on_halt sold both


def test_live_bots_need_permission_and_one_account_per_asset(tmp_path, algos):
    broker = FakeBroker()
    s = BotSet([BotSpec("live", "ETH/USDT", "x", mode="live"), BotSpec("p1", "ETH/USDT", "x", mode="paper"),
                BotSpec("p2", "ETH/EUR", "x", mode="paper"), BotSpec("off", "SOL/USDT", "x", enabled=False)])
    m = manager(s, broker, tmp_path)
    failed = m.start_all()
    assert "live bots run from the terminal" in failed["live"]
    assert "same paper account" in failed["p2"]
    assert m.running() == ["p1"]  # the disabled one isn't started by start_all
    m.start("off")
    assert "off" in m.running()


def test_restart_resumes_the_position(tmp_path, algos):
    broker = FakeBroker()
    s = BotSet([BotSpec("eth", "ETH/USDT", "x", timeframe="1d")])
    m = manager(s, broker, tmp_path)
    m.start("eth")
    algos["eth"].next = 1.0
    m.tick()
    m.shutdown()
    m2 = manager(s, broker, tmp_path)
    m2.start("eth")
    bot = m2._runners["eth"].bot
    assert bot.holding == 1.0 and bot.entry_price == 100.0 and m2.snapshot()[0].events


def test_another_process_owns_a_running_bot(tmp_path, algos):
    s = BotSet([BotSpec("eth", "ETH/USDT", "x", timeframe="1d")])
    state = tmp_path / "state"
    state.mkdir()
    (state / "eth.json").write_text(json.dumps({"name": "eth", "symbol": "ETH/USDT", "mode": "dry", "pid": 1,
                                                "heartbeat": mgr.time.time(), "state": "running", "position": 2,
                                                "entry": 10, "last": 12, "open_pnl": 4, "realized": 1,
                                                "events": [{"time": "2026-11-25T10:00:00", "action": "buy",
                                                            "amount": 2, "price": 10, "note": "", "target": 1}]}))
    m = manager(s, FakeBroker(), tmp_path)
    with pytest.raises(ValueError, match="another process"):
        m.start("eth")
    row = m.snapshot()[0]
    assert row.state == "elsewhere" and row.pnl == 5 and row.last_event == "buy 2 at 10"
    # a stop request from the dashboard reaches the process that runs it
    m.start  # noqa: B018
    own = manager(s, FakeBroker(), tmp_path)
    (state / "eth.json").unlink()
    own.start("eth")
    mgr.request_stop("eth", state)
    own.tick()
    assert own.running() == []


def test_run_loop_stops_when_no_bot_is_left(tmp_path, algos):
    s = BotSet([BotSpec("eth", "ETH/USDT", "x", timeframe="1d")])
    m = manager(s, FakeBroker(), tmp_path)
    m.start("eth")
    mgr.request_stop("eth", tmp_path / "state")
    m.run(poll=0)
    assert m.running() == []


def test_background_loop_steps_and_shuts_down(tmp_path, algos):
    import time

    m = manager(BotSet([BotSpec("eth", "ETH/USDT", "x", timeframe="1d")]), FakeBroker(), tmp_path)
    m.start("eth")
    m.start_loop()
    deadline = time.time() + 5
    while not m.snapshot()[0].events and time.time() < deadline:
        time.sleep(0.02)
    m.shutdown()
    assert m.snapshot()[0].events and m.running() == [] and not m._thread.is_alive()


def test_a_failing_step_is_kept_as_error(tmp_path, algos):
    class Broken(FakeBroker):
        def get_klines(self, *a):
            raise ConnectionError("offline")

    m = manager(BotSet([BotSpec("eth", "ETH/USDT", "x", timeframe="1d")]), Broken(), tmp_path)
    m.start("eth")
    assert m.tick() == []
    row = m.snapshot()[0]
    assert row.state == "error" and "offline" in row.error


def test_reload_stops_removed_and_changed_bots(tmp_path, algos):
    s = BotSet([BotSpec("a", "A/USDT", "x"), BotSpec("b", "B/USDT", "x")])
    m = manager(s, FakeBroker(), tmp_path)
    m.start_all()
    m.reload(BotSet([BotSpec("a", "A/USDT", "x", stake=5)]))
    assert m.names() == ["a"] and m.running() == []


def test_cli_add_list_limits_run_stop(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli.main import app

    broker = FakeBroker()
    monkeypatch.setattr(mgr, "_default_broker", lambda name, sandbox=False: broker)
    r = CliRunner()
    out = r.invoke(app, ["bots", "add", "eth", "ETH/USDT", "sma_cross", "--tf", "1d", "--stake", "50",
                         "-p", "fast=3", "-p", "slow=5", "--max-drawdown", "10"])
    assert out.exit_code == 0, out.output
    assert r.invoke(app, ["bots", "add", "eth", "ETH/USDT", "rsi"]).exit_code != 0  # names are unique
    assert r.invoke(app, ["bots", "add", "x", "ETH/USDT", "no_such_algo"]).exit_code != 0
    assert r.invoke(app, ["bots", "add", "sol", "SOL/USDT", "rsi_threshold", "--mode", "live", "--disabled"]).exit_code == 0
    assert r.invoke(app, ["bots", "limits", "--max-open-positions", "2"]).exit_code == 0
    bots = load_bots()
    assert [b.name for b in bots.bots] == ["eth", "sol"] and bots.bots[0].params == {"fast": 3, "slow": 5}
    assert bots.bots[0].risk == {"max_drawdown": 10} and not bots.bots[1].enabled
    assert bots.limits.max_open_positions == 2
    out = r.invoke(app, ["bots", "run", "--once", "--no-journal"])
    assert out.exit_code == 0, out.output
    assert "started eth" in out.output and "not started" not in out.output  # the disabled live bot isn't tried
    out = r.invoke(app, ["bots", "run", "sol", "--once", "--no-journal"])  # live, without --live
    assert out.exit_code == 1 and "live bots run from the terminal" in " ".join(out.output.split())
    monkeypatch.setattr("tradebot.cli.main.console._width", 200)  # undo restores None, i.e. auto width
    listed = r.invoke(app, ["bots", "list"])
    assert listed.exit_code == 0 and "eth" in listed.output and "stopped" in listed.output
    assert r.invoke(app, ["bots", "stop", "eth"]).exit_code == 0
    assert (mgr.state_dir() / "eth.stop").exists()
    assert r.invoke(app, ["bots", "disable", "eth"]).exit_code == 0 and not load_bots().get("eth").enabled
    assert r.invoke(app, ["bots", "remove", "eth"]).exit_code == 0 and [b.name for b in load_bots().bots] == ["sol"]
    assert r.invoke(app, ["bots", "remove", "nope"]).exit_code != 0
