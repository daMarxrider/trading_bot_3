# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""A running bot survives what goes wrong while it runs: a failing candle step, a failing price tick, an algorithm
whose exits can't be worked out. Offline, with the fake exchange."""

from types import SimpleNamespace

import pytest

from tradebot.live import bot as bot_module
from tradebot.live.bot import exit_levels

from .test_risk import make_bot


@pytest.fixture(autouse=True)
def quiet_notifications(monkeypatch):
    sent = []
    monkeypatch.setattr("tradebot.notify.notify", lambda kind, title, text: sent.append((kind, title)))
    monkeypatch.setattr("tradebot.notify.bot_event", lambda ev: None)
    return sent


def holding_bot():
    bot, algo, ex = make_bot(None)
    algo.next = 1.0
    bot.step()
    return bot


def test_every_event_reaches_the_event_hook():
    bot, algo, _ = make_bot(None)
    seen = []
    bot.on_event = seen.append
    algo.next = 1.0
    ev = bot.step()
    assert seen == [ev]


def test_without_exits_or_limits_a_price_inside_the_candle_does_nothing():
    bot = holding_bot()
    bot.levels = None
    assert bot.on_price(1.0) is None and bot.holding > 0


def test_a_failing_price_tick_is_logged_and_the_stream_goes_on(caplog):
    bot = holding_bot()
    bot.levels = {"stop": 99.0}

    def broken_sell(event, target=0.0):
        raise ConnectionError("exchange down")

    bot._sell = broken_sell
    tick = SimpleNamespace(price=90.0, candle=(0, 100.0, 100.0, 90.0, 90.0))
    bot._on_tick(tick)  # must not raise: it runs on the stream's thread
    assert "intrabar check failed for BTC/USDT" in caplog.text


def test_a_failing_candle_step_is_reported_and_retried_at_the_next_candle(monkeypatch, quiet_notifications, caplog):
    bot, algo, _ = make_bot(None)
    calls = []
    real_step = bot.step

    def flaky_step():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("exchange down")
        return real_step()

    bot.step = flaky_step
    monkeypatch.setattr(bot, "_wait", lambda until, ticks: None)  # no waiting for the next candle
    bot.run(max_steps=2)
    assert len(calls) == 2
    assert quiet_notifications == [("error", "Bot step failed: BTC/USDT")]
    assert "bot step failed for BTC/USDT; retrying next candle" in caplog.text


def test_without_a_stream_the_bot_sleeps_until_the_next_candle(monkeypatch):
    slept = []
    monkeypatch.setattr(bot_module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(bot_module.time, "sleep", slept.append)
    bot, _, _ = make_bot(None)
    bot._wait(1005.0, None)
    assert slept == [5.0]


def test_a_deadline_in_the_past_does_not_wait(monkeypatch):
    slept = []
    monkeypatch.setattr(bot_module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(bot_module.time, "sleep", slept.append)
    bot, _, _ = make_bot(None)
    bot._wait(999.0, None)
    assert slept == []


class BrokenExits:
    def exit_levels(self, df):
        raise ZeroDivisionError("bad ATR")


def test_exits_that_cant_be_worked_out_are_none_and_logged(caplog):
    assert exit_levels(BrokenExits(), None) is None
    assert "couldn't work out the intrabar exits" in caplog.text


class CloseOnly:
    """An algorithm filled at the close only: no intrabar exits."""


def test_an_algorithm_without_intrabar_exits_has_none():
    assert exit_levels(CloseOnly(), None) is None
