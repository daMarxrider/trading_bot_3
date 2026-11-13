# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Live bot risk guards (#154), offline with the fake exchange."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tradebot.brokers.crypto import BinanceBroker
from tradebot.live import RiskGuard, RiskLimits, TradingBot
from tradebot.live import risk

from .test_brokers_data import FakeExchange
from .test_services import Scripted


@pytest.fixture(autouse=True)
def kill_path(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("sys.platform", "linux")


def make_bot(limits, closes=None, stake=100.0, on_risk=None):
    now = pd.Timestamp.now(tz="UTC").floor("1D")
    closes = closes or [100.0] * 9
    start = int((now - pd.Timedelta(days=len(closes))).timestamp() * 1000)
    candles = [[start + i * 86_400_000, c, c, c, c, 1] for i, c in enumerate(closes)]
    ex = FakeExchange(candles)
    algo = Scripted()
    return TradingBot(BinanceBroker(exchange=ex), algo, "BTC/USDT", "1d", stake, dry_run=False, risk=limits,
                      on_risk=on_risk), algo, ex


def test_order_value_and_exposure_are_rejected():
    seen = []
    bot, algo, ex = make_bot(RiskLimits(max_order_value=50), on_risk=seen.append)
    algo.next = 1.0
    ev = bot.step()
    assert ev.action == "hold" and "rejected" in ev.note and not ex.created and bot.holding == 0
    assert seen[0].kind == "order_size" and not seen[0].halts
    bot, algo, ex = make_bot(RiskLimits(max_exposure=99))
    algo.next = 1.0
    assert bot.step().action == "hold" and not ex.created
    bot, algo, ex = make_bot(RiskLimits(max_exposure=100, max_order_value=100))
    algo.next = 1.0
    assert bot.step().action == "buy" and ex.created


def test_price_band_rejects_a_far_off_exchange_price():
    bot, algo, ex = make_bot(RiskLimits(price_band=1))  # fake ticker says 101.5, the close is 100
    algo.next = 1.0
    assert bot.step().action == "hold" and not ex.created
    bot, algo, ex = make_bot(RiskLimits(price_band=2))
    algo.next = 1.0
    assert bot.step().action == "buy"


def test_manual_kill_switch_halts_and_resumes():
    seen = []
    bot, algo, ex = make_bot(RiskLimits(), on_risk=seen.append)
    algo.next = 1.0
    risk.set_kill(True)
    assert risk.kill_active()
    ev = bot.step()
    assert ev.action == "hold" and ev.note.startswith("halted") and not ex.created
    assert [e.kind for e in seen] == ["kill_switch"]
    bot.step()
    assert len(seen) == 1  # triggers once, not on every candle
    risk.set_kill(False)
    bot.guard.resume()
    assert bot.step().action == "buy"


def test_kill_switch_can_close_the_position():
    bot, algo, ex = make_bot(RiskLimits(close_on_halt=True))
    algo.next = 1.0
    bot.step()
    risk.set_kill(True)
    ev = bot.step()
    assert ev.action == "sell" and bot.holding == 0 and ex.created[-1][2] == "sell"


def test_drawdown_halts_on_open_loss():
    bot, algo, ex = make_bot(RiskLimits(max_drawdown=5), stake=100)
    algo.next = 1.0
    bot.step()
    bot.entry_price = 120.0  # the position is now ~17% under water on 100 capital
    ev = bot.step()
    assert ev.note.startswith("halted") and bot.guard.halted.kind == "drawdown"


def test_daily_loss_halts_then_lifts_next_day():
    g = RiskGuard(RiskLimits(max_daily_loss=10), capital=100)
    t0 = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    assert g.check(0, t0) is None
    g.record_trade(-12, t0)
    assert g.check(0, t0).kind == "daily_loss"
    assert g.check(0, t0 + timedelta(hours=1)).kind == "daily_loss"
    assert g.check(0, t0 + timedelta(days=1)) is None


def test_cooldown_after_losses_in_a_row():
    g = RiskGuard(RiskLimits(cooldown_losses=2, cooldown_minutes=30), capital=100)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    g.record_trade(-1, t0)
    g.record_trade(1, t0)  # a win resets the streak
    g.record_trade(-1, t0)
    assert g.check_entry(1, 100, 0, now=t0) is None
    g.record_trade(-1, t0)
    assert g.check_entry(1, 100, 0, now=t0 + timedelta(minutes=10)).kind == "cooldown"
    assert g.check_entry(1, 100, 0, now=t0 + timedelta(minutes=31)) is None


def test_losing_position_trips_the_daily_limit_and_is_closed_in_the_bot():
    bot, algo, ex = make_bot(RiskLimits(max_daily_loss=1, close_on_halt=True))
    algo.next = 1.0
    bot.step()
    bot.entry_price = 110.0  # sold at the fill price 100 -> lost 10 per unit
    assert bot.step().action == "sell"  # open loss 10 > 1: halt, then close_on_halt sells
    assert bot.guard.halted.kind == "daily_loss"
    algo.next = 1.0
    assert bot.step().action == "hold"


def test_no_limits_means_unchanged_behaviour():
    bot, algo, ex = make_bot(None)
    assert bot.guard is None
    algo.next = 1.0
    assert bot.step().action == "buy"


def test_cli_kill_and_resume():
    from typer.testing import CliRunner

    from tradebot.cli.main import app

    r = CliRunner()
    assert r.invoke(app, ["bot", "kill"]).exit_code == 0 and risk.kill_active()
    assert r.invoke(app, ["bot", "resume"]).exit_code == 0 and not risk.kill_active()
