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


# -- RiskGuard on its own: boundaries, broken numbers, the trigger hook (clock and kill switch injected) ----
T0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)


def guard(capital=1000.0, on_trigger=None, killed=False, **limits):
    return RiskGuard(RiskLimits(**limits), capital, on_trigger=on_trigger, kill_path=lambda: killed)


def test_daily_loss_exactly_at_the_limit_halts():
    g = guard(max_daily_loss=50)
    g.check(0.0, T0)
    assert g.check(-50.0, T0).kind == "daily_loss"


def test_daily_loss_just_under_the_limit_does_not_halt():
    g = guard(max_daily_loss=50)
    g.check(0.0, T0)
    assert g.check(-49.99, T0) is None


def test_daily_loss_counts_from_the_start_of_the_day_not_from_zero():
    g = guard(max_daily_loss=50)
    g.check(-200.0, T0)  # yesterday's losses are not today's
    assert g.check(-240.0, T0 + timedelta(hours=1)) is None


def test_a_drawdown_halt_does_not_lift_the_next_day():
    g = guard(max_drawdown=10)
    g.check(-100.0, T0)  # 100 of 1000 = 10 %
    assert g.check(0.0, T0 + timedelta(days=1)).kind == "drawdown"


def test_drawdown_uses_the_configured_capital_over_the_stake():
    g = RiskGuard(RiskLimits(max_drawdown=10, capital=100.0), 1000.0, kill_path=lambda: False)
    assert g.check(-10.0, T0).kind == "drawdown"  # 10 of 100, not of the 1000 stake


def test_drawdown_with_zero_capital_is_switched_off_not_a_division_error():
    g = guard(capital=0.0, max_drawdown=10)
    assert g.check(-50.0, T0) is None


def test_after_resume_drawdown_is_measured_from_the_current_level():
    g = guard(max_drawdown=10)
    g.record_trade(-100.0, T0)
    g.check(0.0, T0)
    g.resume()
    assert g.check(-50.0, T0) is None  # 50 below the level at resume: 5 %


def test_a_halt_fires_the_hook_once_while_it_lasts():
    seen = []
    g = guard(on_trigger=seen.append, max_daily_loss=10)
    g.check(-20.0, T0)
    g.check(-30.0, T0)
    assert [e.kind for e in seen] == ["daily_loss"]


def test_a_failing_hook_does_not_stop_the_halt(caplog):
    def broken(_event):
        raise RuntimeError("telegram is down")

    g = guard(on_trigger=broken, killed=True)
    assert g.check(0.0, T0).kind == "kill_switch"
    assert "risk trigger hook failed" in caplog.text


def test_the_cooldown_needs_losses_in_a_row():
    g = guard(cooldown_losses=2, cooldown_minutes=30)
    g.record_trade(-1.0, T0)
    g.record_trade(+1.0, T0)  # a win resets the run
    g.record_trade(-1.0, T0)
    assert g.check_entry(1, 10, 0, now=T0) is None


def test_the_cooldown_blocks_entries_until_it_ends():
    g = guard(cooldown_losses=2, cooldown_minutes=30)
    g.record_trade(-1.0, T0)
    g.record_trade(-1.0, T0)
    assert g.check_entry(1, 10, 0, now=T0 + timedelta(minutes=29)).kind == "cooldown"


def test_entries_are_allowed_again_when_the_cooldown_ends():
    g = guard(cooldown_losses=2, cooldown_minutes=30)
    g.record_trade(-1.0, T0)
    g.record_trade(-1.0, T0)
    assert g.check_entry(1, 10, 0, now=T0 + timedelta(minutes=30)) is None


def test_a_zero_cooldown_count_never_cools_down():
    g = guard(cooldown_losses=0, cooldown_minutes=30)
    for _ in range(10):
        g.record_trade(-1.0, T0)
    assert g.check_entry(1, 10, 0, now=T0) is None


def test_an_order_exactly_at_the_value_cap_is_allowed():
    assert guard(max_order_value=100).check_entry(2, 50.0, 0, now=T0) is None


def test_exposure_includes_what_is_already_held():
    # holding 1 at 50 + buying 1 at 50 = 100 > 99
    assert guard(max_exposure=99).check_entry(1, 50.0, held=1, now=T0).kind == "exposure"


@pytest.mark.parametrize("amount, price", [(float("nan"), 50.0), (1.0, float("nan")), (float("inf"), 50.0)])
def test_an_order_whose_value_cant_be_computed_is_rejected(amount, price):
    ev = guard(max_order_value=100).check_entry(amount, price, 0, now=T0)
    assert ev is not None and ev.kind == "order_size" and not ev.halts


def test_a_nan_exchange_price_fails_the_price_band():
    assert guard(price_band=1).check_entry(1, 100.0, 0, market_price=float("nan"), now=T0).kind == "price_band"


def test_no_exchange_price_skips_the_price_band():
    assert guard(price_band=1).check_entry(1, 100.0, 0, market_price=None, now=T0) is None


def test_a_zero_close_skips_the_price_band_instead_of_dividing_by_zero():
    assert guard(price_band=1).check_entry(1, 0.0, 0, market_price=50.0, now=T0) is None


def test_the_kill_file_round_trip():
    risk.set_kill(True, reason="test")
    on = risk.kill_active()
    risk.set_kill(False)
    assert on and not risk.kill_active()


def test_switching_off_a_kill_switch_that_is_not_on_is_fine():
    risk.set_kill(False)
    assert not risk.kill_active()


# -- bad price data (a NaN profit): an explicit, self-lifting halt that leaves every limit intact ----------------
NAN = float("nan")


def test_a_nan_open_profit_halts_as_bad_data():
    ev = guard(max_daily_loss=50).check(NAN, T0)
    assert (ev.kind, ev.halts) == ("bad_data", True)


def test_bad_data_halts_even_without_any_limit_set():
    assert guard().check(float("inf"), T0).kind == "bad_data"


def test_bad_data_fires_the_hook_once_while_it_lasts():
    seen = []
    g = guard(on_trigger=seen.append)
    g.check(NAN, T0)
    g.check(NAN, T0)
    assert [e.kind for e in seen] == ["bad_data"]


def test_the_bad_data_halt_lifts_at_the_first_valid_profit():
    g = guard(max_daily_loss=50)
    g.check(NAN, T0)
    assert g.check(-10.0, T0) is None


def test_bad_data_at_the_start_of_a_day_does_not_switch_off_the_daily_limit():
    g = guard(max_daily_loss=50)
    g.check(0.0, T0)
    next_day = T0 + timedelta(days=1)
    g.check(NAN, next_day)  # the new day's first check: must not become its starting point (that turned it off)
    g.check(0.0, next_day + timedelta(minutes=1))  # the day starts here, at 0
    assert g.check(-50.0, next_day + timedelta(minutes=2)).kind == "daily_loss"


def test_bad_data_does_not_reset_the_drawdown_peak():
    g = guard(max_drawdown=10)
    g.check(100.0, T0)  # peak 100
    g.check(NAN, T0)
    assert g.check(0.0, T0).kind == "drawdown"  # 100 below the peak = 10 % of 1000


def test_bad_data_does_not_lift_a_lasting_halt():
    g = guard(max_drawdown=10)
    g.check(-100.0, T0)
    g.check(NAN, T0)
    assert g.check(0.0, T0).kind == "drawdown"


def test_a_lasting_halt_is_reported_over_bad_data():
    g = guard(killed=True)
    g.check(0.0, T0)
    assert g.check(NAN, T0).kind == "kill_switch"


def test_a_nan_trade_result_is_not_counted():
    g = guard(max_daily_loss=50)
    g.record_trade(-20.0, T0)
    g.record_trade(NAN, T0)
    assert g.realized == -20.0


def test_a_nan_trade_result_is_reported():
    seen = []
    guard(on_trigger=seen.append).record_trade(NAN, T0)
    assert [(e.kind, e.halts) for e in seen] == [("bad_data", False)]


def test_a_nan_trade_result_does_not_break_the_losing_streak_count():
    g = guard(cooldown_losses=2, cooldown_minutes=30)
    g.record_trade(-1.0, T0)
    g.record_trade(NAN, T0)
    g.record_trade(-1.0, T0)
    assert g.check_entry(1, 10, 0, now=T0).kind == "cooldown"


def test_the_bot_does_not_buy_while_prices_are_bad():
    bot, algo, ex = make_bot(RiskLimits())
    bot.guard.realized = NAN  # as if a fill had been recorded at a NaN price
    algo.next = 1.0
    ev = bot.step()
    assert ev.action == "hold" and "bad price data" in ev.note and not ex.created


def test_close_on_halt_does_not_sell_on_bad_data():
    bot, algo, ex = make_bot(RiskLimits(close_on_halt=True))
    algo.next = 1.0
    bot.step()
    bot.entry_price = NAN  # the open profit can't be computed
    ev = bot.step()
    assert ev.action == "hold" and bot.holding > 0 and [o[2] for o in ex.created] == ["buy"]


def test_the_bot_trades_again_once_prices_are_valid():
    bot, algo, ex = make_bot(RiskLimits(close_on_halt=True))
    algo.next = 1.0
    bot.step()
    bot.entry_price, keep = NAN, bot.entry_price
    bot.step()
    bot.entry_price = keep
    algo.next = 0.0
    assert bot.step().action == "sell"


def test_close_on_halt_inside_the_candle_ignores_bad_data():
    bot, algo, ex = make_bot(RiskLimits(close_on_halt=True))
    algo.next = 1.0
    bot.step()
    bot.entry_price = NAN
    assert bot.on_price(90.0) is None and bot.holding > 0


@pytest.mark.parametrize("tick", [NAN, float("inf")])
def test_a_broken_stream_tick_is_ignored(tick):
    bot, algo, ex = make_bot(RiskLimits(close_on_halt=True, max_daily_loss=1))
    algo.next = 1.0
    bot.step()
    assert bot.on_price(tick) is None and bot.holding > 0 and bot.guard.halted is None
