# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The live pair bot remembers its open pair across restarts and survives partial fills and failed undos (#122)."""

import json

import pytest

import tradebot.live.pairs as lp
from tradebot.brokers.base import BaseBroker
from tradebot.core.models import Order, OrderStatus, OrderType, Ticker
from tradebot.live import PairBot, PairStateError

from .test_pairs import StubData, synthetic_pair

LEGS = ("AAA/USDT:USDT", "BBB/USDT:USDT")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(lp.notify, "notify", lambda *a, **k: 0)


@pytest.fixture
def signal(monkeypatch):
    """Drive the bot by hand: the side it wants next."""
    box = {"side": 0.0}
    monkeypatch.setattr(lp, "next_side", lambda s, *a: box["side"])
    return box


class Futures(BaseBroker):
    """A futures account: signed positions, scripted fills and failures."""

    name = "fake-futures"

    def __init__(self, data):
        super().__init__(data=data)
        self.pos = {s: 0.0 for s in LEGS}
        self.orders, self.fail, self.fill_ratio, self.open_orders, self.can_lookup = [], {}, {}, {}, True

    def get_ticker(self, symbol):
        return Ticker(symbol, 1.0)

    def place_order(self, symbol, side, amount, type=OrderType.MARKET, price=None):
        exc = self.fail.pop((symbol, side.value), None)
        if exc is not None:
            raise exc
        ratio = self.fill_ratio.pop((symbol, side.value), 1.0)
        filled = amount * ratio
        self.pos[symbol] += filled if side.value == "buy" else -filled
        oid = str(len(self.orders) + 1)
        self.orders.append((symbol, side.value, amount, filled))
        status = OrderStatus.CLOSED if ratio == 1.0 else OrderStatus.OPEN
        o = Order(oid, symbol, side, type, amount, None, status, filled)
        if status is OrderStatus.OPEN:
            self.open_orders[oid] = o
        return o

    def get_order(self, order_id, symbol=None):
        return self.open_orders.get(order_id) if self.can_lookup else None

    def cancel_order(self, order_id, symbol=None):
        self.open_orders[order_id].status = OrderStatus.CANCELED
        return True

    def get_balance(self):
        return {}

    def leg_position(self, symbol):
        return self.pos[symbol]


@pytest.fixture
def broker():
    a, b = synthetic_pair(n=120, seed=11)
    return Futures(StubData(a, b))


def make(broker, path, **kw):
    return PairBot(broker, LEGS, "1d", stake=1000.0, window=20, dry_run=False, state_path=path,
                   **{"poll_seconds": 0, "fill_timeout": 0, **kw})


def saved(path):
    return json.loads(path.read_text())


def test_open_pair_survives_a_restart_and_is_closed_with_the_saved_amounts(tmp_path, broker, signal):
    path = tmp_path / "pair.json"
    bot = make(broker, path)
    signal["side"] = 1.0
    e = bot.step()
    assert e.action == "open long" and bot.side == 1.0
    s = saved(path)
    assert s["side"] == 1.0 and s["pending"] is None and s["amounts"] == list(bot.amounts) and s["opened"]
    assert broker.pos[LEGS[0]] == pytest.approx(bot.amounts[0]) and broker.pos[LEGS[1]] == pytest.approx(-bot.amounts[1])
    assert not list(tmp_path.glob("*.tmp"))  # atomic write leaves no temp files behind

    again = make(broker, path)  # restart
    assert again.side == 1.0 and again.amounts == bot.amounts
    assert again.reconcile() == []
    signal["side"] = 0.0
    assert again.step().action == "close"
    assert broker.pos == {LEGS[0]: pytest.approx(0.0), LEGS[1]: pytest.approx(0.0)}
    assert broker.orders[-2][:2] == (LEGS[1], "buy") and broker.orders[-1][:2] == (LEGS[0], "sell")
    assert saved(path)["side"] == 0.0 and make(broker, path).side == 0.0


def test_restart_after_a_crash_mid_trade_refuses_until_checked(tmp_path, broker, signal):
    from typer.testing import CliRunner

    from tradebot.cli.main import app

    path = lp.pair_state_path("fake", LEGS)
    bot = make(broker, path)
    broker.fail[(LEGS[1], "sell")] = KeyboardInterrupt()  # the process dies between the two legs
    signal["side"] = 1.0
    with pytest.raises(KeyboardInterrupt):
        bot.step()
    p = saved(path)["pending"]
    assert p["action"] == "open long" and p["orders"][0]["filled"] > 0 and p["orders"][1]["filled"] is None
    with pytest.raises(PairStateError, match="did not finish"):
        make(broker, path)

    r = CliRunner()
    out = r.invoke(app, ["bot", "pair-state", *LEGS, "--broker", "fake"])
    text = " ".join(out.output.split())  # rich wraps at the terminal width
    assert out.exit_code == 0 and "unfinished trade" in text and "may have been placed" in text
    assert r.invoke(app, ["bot", "pair-state", *LEGS, "--broker", "fake", "--keep", "--forget"]).exit_code != 0
    assert r.invoke(app, ["bot", "pair-state", *LEGS, "--broker", "fake", "--keep", "-y"]).exit_code == 0
    assert make(broker, path).side == 0.0  # the last finished state: flat


def test_unreadable_or_foreign_state_is_refused(tmp_path, broker):
    path = tmp_path / "pair.json"
    path.write_text("{not json")
    with pytest.raises(PairStateError, match="unreadable"):
        make(broker, path)
    path.write_text(json.dumps({"version": 1, "legs": ["X", "Y"], "side": 0.0, "amounts": [0, 0], "pending": None}))
    with pytest.raises(PairStateError, match="belongs to"):
        make(broker, path)
    path.write_text(json.dumps({"version": 1, "legs": list(LEGS), "side": 2.0, "amounts": [0, 0]}))
    with pytest.raises(PairStateError):
        make(broker, path)
    with pytest.raises(ValueError):  # a dry run never writes the live state
        PairBot(broker, LEGS, dry_run=True, state_path=path)


def test_cli_forget_fixes_a_corrupt_file(broker):
    from typer.testing import CliRunner

    from tradebot.cli.main import app

    path = lp.pair_state_path("fake", LEGS, sandbox=True)
    assert path != lp.pair_state_path("fake", LEGS) and "/" not in path.name and ":" not in path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("garbage")
    r = CliRunner()
    assert r.invoke(app, ["bot", "pair-state", *LEGS, "--broker", "fake", "--sandbox"]).exit_code == 1
    assert r.invoke(app, ["bot", "pair-state", *LEGS, "--broker", "fake", "--sandbox", "--forget", "-y"]).exit_code == 0
    assert make(broker, path).side == 0.0


def test_partial_first_leg_shrinks_the_second_and_amounts_are_the_fills(tmp_path, broker, signal):
    path = tmp_path / "pair.json"
    bot = make(broker, path)
    broker.fill_ratio[(LEGS[0], "buy")] = 0.5  # stays open, the rest is cancelled after the timeout
    signal["side"] = 1.0
    e = bot.step()
    (_, _, want_a, got_a), (_, _, sell_b, got_b) = broker.orders
    assert got_a == pytest.approx(want_a / 2) and e.note.startswith("partly filled")
    assert bot.amounts == (pytest.approx(got_a), pytest.approx(got_b))
    assert sell_b == pytest.approx(e.amount_b) and broker.open_orders["1"].status is OrderStatus.CANCELED
    assert saved(path)["amounts"] == [pytest.approx(got_a), pytest.approx(got_b)]


def test_partial_close_keeps_closing_the_rest_whatever_the_signal(tmp_path, broker, signal):
    path = tmp_path / "pair.json"
    bot = make(broker, path)
    signal["side"] = -1.0  # short the gap: long B, short A
    bot.step()
    amt_a, amt_b = bot.amounts
    broker.fail[(LEGS[1], "sell")] = RuntimeError("rejected")  # A bought back, selling B fails
    signal["side"] = 0.0
    with pytest.raises(RuntimeError, match="rejected"):
        bot.step()
    assert bot.side == -1.0 and bot.closing and bot.amounts == (0.0, pytest.approx(amt_b))
    restarted = make(broker, path)
    assert restarted.closing and restarted.reconcile() == []
    signal["side"] = -1.0  # the signal says hold, but half a pair is no pair: close the rest
    assert restarted.step().action == "close"
    assert restarted.side == 0.0 and not restarted.closing and broker.pos[LEGS[1]] == pytest.approx(0.0)


def test_rejected_short_is_undone_and_a_failed_undo_halts(tmp_path, broker, signal):
    path = tmp_path / "pair.json"
    bot = make(broker, path)
    broker.fail[(LEGS[1], "sell")] = RuntimeError("can't short")
    signal["side"] = 1.0
    with pytest.raises(RuntimeError, match="can't short"):
        bot.step()
    assert bot.side == 0.0 and broker.pos[LEGS[0]] == pytest.approx(0.0) and saved(path)["pending"] is None

    broker.fail[(LEGS[1], "sell")] = RuntimeError("can't short")
    broker.fail[(LEGS[0], "sell")] = RuntimeError("exchange down")  # the undo fails as well
    with pytest.raises(PairStateError, match="undoing the purchase failed"):
        bot.step()
    assert bot.halted and saved(path)["pending"]["problem"]
    with pytest.raises(PairStateError, match="halted"):
        bot.step()
    with pytest.raises(PairStateError, match="undoing"):
        make(broker, path)


def test_a_timeout_while_placing_halts_because_the_order_may_exist(tmp_path, broker, signal):
    path = tmp_path / "pair.json"
    bot = make(broker, path)
    broker.fail[(LEGS[0], "buy")] = TimeoutError("read timed out")
    signal["side"] = 1.0
    with pytest.raises(PairStateError, match="may have gone through"):
        bot.step()
    assert saved(path)["pending"]["orders"][0]["filled"] is None

    other = make(broker, tmp_path / "other.json")  # a plain rejection is no doubt: nothing is pending
    broker.fail[(LEGS[0], "buy")] = RuntimeError("insufficient margin")
    with pytest.raises(RuntimeError, match="margin"):
        other.step()
    assert saved(tmp_path / "other.json")["pending"] is None and other.side == 0.0


def test_open_order_that_cant_be_looked_up_halts(tmp_path, broker, signal):
    path = tmp_path / "pair.json"
    bot = make(broker, path, fill_timeout=60)
    broker.can_lookup = False
    broker.fill_ratio[(LEGS[0], "buy")] = 0.0  # e.g. a market order reported as "open, filled 0"
    signal["side"] = 1.0
    with pytest.raises(PairStateError, match="can't look it up"):
        bot.step()
    assert bot.halted and bot.side == 0.0 and len(broker.orders) == 1  # no second leg on a guess
    p = saved(path)["pending"]
    assert p["problem"] and p["orders"][0]["id"] == "1" and p["orders"][0]["filled"] is None


def test_orders_go_to_the_journal_with_what_filled(tmp_path, broker, signal):
    from tradebot.journal import Journal

    j = Journal(tmp_path / "j.sqlite")
    bot = make(broker, tmp_path / "pair.json", journal=j)
    broker.fill_ratio[(LEGS[0], "buy")] = 0.5
    signal["side"] = 1.0
    bot.step()
    fills = j.fills()
    assert [f["symbol"] for f in fills] == list(LEGS)
    assert fills[0]["amount"] == pytest.approx(broker.orders[0][3])


def test_reconcile_refuses_when_the_account_lost_the_pair(tmp_path, broker, signal):
    path = tmp_path / "pair.json"
    bot = make(broker, path)
    signal["side"] = 1.0
    bot.step()
    broker.pos[LEGS[1]] = 0.0  # someone closed the short leg by hand
    with pytest.raises(PairStateError, match="doesn't match"):
        make(broker, path).run(max_steps=1)
    broker.pos[LEGS[1]] = -bot.amounts[1] * 0.995  # fees: a little less is fine
    assert make(broker, path).reconcile() == []
    broker.leg_position = lambda symbol: None  # a broker that can't report positions trusts the file
    assert len(make(broker, path).reconcile()) == 2


def test_ccxt_futures_positions_and_open_market_orders():
    """Binance futures answers a market order with status "open" and filled 0; the fill shows in fetch_order."""
    from tradebot.brokers.crypto import BinanceBroker

    class Ex:
        has = {"fetchOrder": True, "fetchPositions": True}

        def __init__(self):
            self.orders = {}

        def load_markets(self):
            pass

        def market(self, symbol):
            return {"symbol": symbol, "base": symbol.split("/")[0], "contract": symbol.endswith(":USDT"),
                    "contractSize": 1}

        def amount_to_precision(self, symbol, amount):
            return f"{amount:.3f}"

        def create_order(self, symbol, type, side, amount, price):
            self.orders["7"] = {"id": "7", "symbol": symbol, "side": side, "type": type, "amount": amount,
                                "status": "closed", "filled": amount, "average": 1.0}
            return {"id": "7", "symbol": symbol, "status": "open", "amount": amount, "filled": 0}

        def fetch_order(self, oid, symbol):
            return self.orders[oid]

        def fetch_positions(self, symbols):
            return [{"symbol": "BTC/USDT:USDT", "contracts": 0.5, "contractSize": 1, "side": "short"},
                    {"symbol": "ETH/USDT:USDT", "contracts": 2, "side": "long"}]

        def fetch_balance(self):
            return {"free": {"BTC": 0.25}, "used": {}}

    b = BinanceBroker(exchange=Ex())
    assert b.leg_position("BTC/USDT:USDT") == -0.5 and b.leg_position("BTC/USDT") == 0.25
    o = b.place_order("BTC/USDT:USDT", "sell", 0.1234)
    assert o.status is OrderStatus.OPEN and o.filled == 0
    got = b.get_order(o.id, o.symbol)
    assert got.status is OrderStatus.CLOSED and got.filled == pytest.approx(0.123) and got.side.value == "sell"
