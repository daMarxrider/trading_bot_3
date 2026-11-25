# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Streaming market data (#158): websocket clients, reconnects, the polling fallback and the bot acting inside
the candle. Only fake clients and a local fake websocket server: no network."""

import asyncio
import base64
import json
import threading
import time

import pandas as pd
import pytest

from tradebot.algorithms.combo import check
from tradebot.backtest import backtest
from tradebot.brokers.crypto import BinanceBroker
from tradebot.data import stream as st
from tradebot.data.stream import MarketStream, Tick, YahooStreamClient, candle_start, open_stream
from tradebot.live import RiskLimits, TradingBot
from tradebot.live.bot import exit_levels

from .test_brokers_data import FakeExchange
from .test_services import Scripted

H = 3_600_000


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


class FakeOHLCVClient:
    """ccxt pro's interface with scripted candle updates; ``fail`` sessions raise before any data."""

    has = {"watchOHLCV": True, "watchTicker": True}

    def __init__(self, updates, log):
        self.updates, self.log, self.closed = list(updates), log, False
        log.append(self)

    async def watch_ohlcv(self, symbol, timeframe):
        if not self.updates:
            await asyncio.sleep(3600)  # nothing more: block like a quiet socket until cancelled
        item = self.updates.pop(0)
        if isinstance(item, Exception):
            raise item
        await asyncio.sleep(0)
        return item

    async def close(self):
        self.closed = True


class FakeTickerClient:
    has = {"watchTicker": True}

    def __init__(self, prices):
        self.prices = list(prices)

    async def watch_ticker(self, symbol):
        if not self.prices:
            await asyncio.sleep(3600)
        price, ts = self.prices.pop(0)
        return {"last": price, "timestamp": ts}

    async def close(self):
        pass


def collect(stream):
    ticks, states = [], []
    stream.on_tick(ticks.append)
    stream.on_state(lambda s, note: states.append(s))
    return ticks, states


def test_candle_start():
    ts = pd.Timestamp("2026-11-25 13:47:12", tz="UTC").timestamp() * 1000
    assert candle_start(ts, "1h") == pd.Timestamp("2026-11-25 13:00", tz="UTC").timestamp() * 1000
    assert candle_start(ts, "4h") == pd.Timestamp("2026-11-25 12:00", tz="UTC").timestamp() * 1000
    assert candle_start(ts, "1w") == pd.Timestamp("2026-11-23", tz="UTC").timestamp() * 1000  # a Monday
    assert candle_start(ts, "1mo") == pd.Timestamp("2026-11-01", tz="UTC").timestamp() * 1000


def test_websocket_candles_become_ticks_in_order():
    clients = []
    updates = [[[0, 10, 11, 9, 10.5, 1]], [[0, 10, 12, 9, 11, 2]], [[0, 10, 12, 9, 11.5, 3], [H, 11.5, 11.5, 11, 11, 1]],
               [[0, 1, 1, 1, 1, 1]]]  # the last one is older than the forming candle: dropped
    s = MarketStream("BTC/USDT", "1h", lambda: FakeOHLCVClient(updates, clients))
    ticks, states = collect(s)
    with s:
        assert wait_for(lambda: len(ticks) >= 4)
        time.sleep(0.05)
        assert s.state == "streaming"
    assert [t.price for t in ticks] == [10.5, 11, 11.5, 11]
    assert ticks[2].candle[0] == pd.Timestamp(0, tz="UTC") and ticks[3].candle[0] == pd.Timestamp(H, unit="ms", tz="UTC")
    assert ticks[1].candle[1:5] == (10, 12, 9, 11) and all(t.source == "websocket" for t in ticks)
    assert states[:2] == ["connecting", "streaming"] and states[-1] == "stopped"
    assert s.state == "stopped" and clients[0].closed and not s.running


def test_ticker_prices_extend_the_polled_snapshot():
    snapshot = pd.DataFrame({"open": [100.0], "high": [105.0], "low": [99.0], "close": [101.0], "volume": [7.0]},
                            index=[pd.Timestamp(0, tz="UTC")])
    prices = [(103.0, 1000), (98.0, 2000), (104.0, H + 5)]  # the third opens the next candle
    s = MarketStream("BTC/USDT", "1h", lambda: FakeTickerClient(prices), lambda: snapshot)
    ticks, _ = collect(s)
    with s:
        assert wait_for(lambda: len(ticks) >= 4)
    assert ticks[0].candle[1:] == (100, 105, 99, 101, 7) and ticks[0].source == "polling"
    assert ticks[2].candle[1:5] == (100, 105, 98, 98)
    assert ticks[3].candle == (pd.Timestamp(H, unit="ms", tz="UTC"), 104, 104, 104, 104, 0)


def test_reconnects_after_a_drop():
    clients, sessions = [], iter([[ConnectionError("dropped")], [ConnectionError("refused")], [[[0, 1, 1, 1, 1, 1]]]])
    s = MarketStream("BTC/USDT", "1h", lambda: FakeOHLCVClient(next(sessions), clients), backoff=0.01, retries=5)
    ticks, states = collect(s)
    with s:
        assert wait_for(lambda: ticks)
    assert len(clients) == 3 and all(c.closed for c in clients)
    assert "reconnecting" in states and states.index("reconnecting") < states.index("streaming")


def test_falls_back_to_polling_and_tries_the_websocket_again():
    clients, polls = [], []
    frame = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]}, index=[pd.Timestamp(0, tz="UTC")])

    def poll():
        polls.append(1)
        return frame

    s = MarketStream("BTC/USDT", "1h", lambda: FakeOHLCVClient([ConnectionError("down")], clients), poll,
                     backoff=0.01, retries=2, poll_interval=0.01, retry_after=0.3)
    ticks, states = collect(s)
    with s:
        assert wait_for(lambda: s.state == "polling")
        assert "down" in s.note
        assert wait_for(lambda: ticks and ticks[0].source == "polling")
        assert wait_for(lambda: len(clients) >= 3)  # retried after retry_after
    assert polls and ticks[0].candle[5] == 0.0  # no volume column: 0


def test_polling_only_and_a_failing_poll_keeps_going():
    calls = []

    def poll():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("timeout")
        return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [len(calls) * 1.0]},
                            index=[pd.Timestamp(0, tz="UTC")])

    s = MarketStream("AAPL", "1d", None, poll, poll_interval=0.01)
    ticks, states = collect(s)
    with s:
        assert wait_for(lambda: len(ticks) >= 2)
        assert s.state == "polling"
    assert ticks[0].price == 2.0 and ticks[1].price == 3.0
    with pytest.raises(ValueError):
        MarketStream("AAPL", "1d")


def test_stop_cancels_a_quiet_socket_quickly():
    clients = []
    s = MarketStream("BTC/USDT", "1h", lambda: FakeOHLCVClient([], clients)).start()
    assert wait_for(lambda: clients)
    t = time.monotonic()
    s.stop()
    assert time.monotonic() - t < 2 and clients[0].closed and not s.running


def test_open_stream_picks_the_source():
    ex = FakeExchange([[0, 1, 1, 1, 1, 1]])
    s = open_stream("BTC/USDT", "1h", broker=BinanceBroker(exchange=ex))
    assert s.client_factory is not None and s.poll().iloc[-1]["close"] == 1  # ccxt pro binance, broker candles
    s = open_stream("BTC/USDT", "1h", broker=BinanceBroker(exchange=ex), websocket=False)
    assert s.client_factory is None
    assert open_stream("AAPL", "1d", engine=type("E", (), {"providers": []})()).client_factory is YahooStreamClient
    assert st.ccxt_pro_factory("no_such_exchange") is None
    with pytest.raises(ValueError):
        open_stream("AAPL", "1d", engine=type("E", (), {"providers": []})(), websocket=False)


def _yahoo_message(symbol, price, ms):
    from yfinance.pricing_pb2 import PricingData

    return json.dumps({"message": base64.b64encode(PricingData(id=symbol, price=price, time=ms).SerializeToString())
                       .decode()})


def test_yahoo_stream_against_a_local_fake_server():
    """The Yahoo client against a websocket server on localhost speaking Yahoo's message format."""
    from websockets.asyncio.server import serve

    subscribed, ready, port = [], threading.Event(), []
    stop = threading.Event()

    async def handler(ws):
        subscribed.append(json.loads(await ws.recv()))
        for i, price in enumerate((190.5, 191.25)):
            await ws.send(_yahoo_message("MSFT", 1.0, 1_000))  # another symbol: skipped
            await ws.send(_yahoo_message("AAPL", price, 1_000 + i))
        await asyncio.sleep(0.2)  # then the server closes the connection

    async def main():
        async with serve(handler, "127.0.0.1", 0) as server:
            port.append(server.sockets[0].getsockname()[1])
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.02)

    server = threading.Thread(target=asyncio.run, args=(main(),), daemon=True)
    server.start()
    assert ready.wait(5)
    url = f"ws://127.0.0.1:{port[0]}"
    s = MarketStream("AAPL", "1d", lambda: YahooStreamClient(url),
                     backoff=0.05, retries=10)
    ticks, states = collect(s)
    try:
        with s:
            assert wait_for(lambda: len(ticks) >= 2)
            assert wait_for(lambda: "reconnecting" in states)  # the server hung up: the stream reconnects
    finally:
        stop.set()
        server.join(5)
    assert subscribed[0] == {"subscribe": ["AAPL"]}
    assert [t.price for t in ticks[:2]] == [190.5, 191.25] and ticks[0].candle[0] == pd.Timestamp(0, tz="UTC")


# -- the bot acting inside the candle ------------------------------------------------------------------------


def frame(closes, end=None):
    end = end or pd.Timestamp.now(tz="UTC").floor("1D")
    idx = pd.date_range(end=end - pd.Timedelta(days=1), periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0},
                        index=idx, dtype=float)


def test_combo_exit_levels_are_the_backtest_levels():
    df = frame([100.0, 100, 104, 102])
    combo = check("close > 0", "stop_pct(5) | take_profit(10, 0.5) | take_profit(20)", "intrabar")
    lv = combo.exit_levels(df)
    assert lv["position"] == 1 and lv["stop"] == pytest.approx(95)  # entered at the first close (100)
    assert lv["take_profits"] == [(pytest.approx(110), 0.5), (pytest.approx(120), 1.0)]
    # the next candle dipping to the stop is sold there in the backtest
    nxt = pd.concat([df, pd.DataFrame({"open": [101.0], "high": [101.0], "low": [94.0], "close": [99.0], "volume": 1.0},
                                      index=[df.index[-1] + pd.Timedelta(days=1)])])
    assert backtest(nxt, combo, fee=0.0).positions.iloc[-1] == 0
    trail = check("close > 0", "trail_pct(10)", "intrabar").exit_levels(df)
    assert trail["stop"] == pytest.approx(104 * 0.9) and trail["take_profits"] == []
    assert check("close > 0", "stop_pct(5)", "close").exit_levels(df) is None
    assert check("close > 0", "stop_pct(5)", "intrabar").exit_levels(frame([100.0])) is not None
    assert check("close < 0", "stop_pct(5)", "intrabar").exit_levels(df) is None  # flat


def stream_bot(exit, dry_run=False, risk=None, sizing=None):
    from tradebot.algorithms.sizing import sized

    df = frame([100.0] * 9)
    candles = [[int(t.timestamp() * 1000), 100, 100, 100, 100, 1] for t in df.index]
    ex = FakeExchange(candles)
    algo = check("close > 0", exit, "intrabar")
    if sizing:
        algo = sized(algo, sizing)
    bot = TradingBot(BinanceBroker(exchange=ex), algo, "BTC/USDT", "1d", 100.0, dry_run=dry_run, risk=risk,
                     stream=object())
    return bot, ex


def test_bot_sells_on_a_stop_inside_the_candle():
    bot, ex = stream_bot("stop_pct(5)", dry_run=True)
    assert bot.step().action == "buy" and bot.levels["stop"] == pytest.approx(95)
    assert bot.on_price(97.0) is None
    ev = bot.on_price(96.0, low=94.5)  # the candle's low went through the stop between two updates
    assert ev.action == "sell" and ev.amount == pytest.approx(1.0) and "stop 95" in ev.note and bot.holding == 0
    assert bot.on_price(90.0) is None and bot.events[-1] is ev


def test_bot_scales_out_on_take_profits_inside_the_candle():
    bot, ex = stream_bot("take_profit(5, 0.5) | take_profit(10)", dry_run=True)
    bot.step()
    ev = bot.on_price(106.0)
    assert ev.action == "sell" and ev.amount == pytest.approx(0.5) and bot.fraction == pytest.approx(0.5)
    assert "take-profit 105" in ev.note
    assert bot.on_price(107.0) is None  # that level is done
    ev = bot.on_price(111.0)
    assert ev.amount == pytest.approx(0.5) and bot.holding == pytest.approx(0)


def test_sized_combo_levels_and_risk_halt_inside_the_candle(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr("sys.platform", "linux")
    bot, ex = stream_bot("stop_pct(50)", sizing="fixed")
    assert exit_levels(bot.algorithm, frame([100.0] * 9))["stop"] == pytest.approx(50)
    bot, ex = stream_bot("stop_pct(50)", risk=RiskLimits(max_drawdown=10, close_on_halt=True))
    bot.step()
    assert ex.created[-1][2] == "buy"
    assert bot.on_price(95.0) is None
    ev = bot.on_price(80.0)  # 20% of the stake below the peak: halted, sold now instead of at the close
    assert ev.action == "sell" and "halted inside the candle" in ev.note and ex.created[-1][2] == "sell"


def test_run_handles_stream_ticks_between_closes(monkeypatch):
    bot, _ = stream_bot("stop_pct(5)", dry_run=True)

    class FakeStream:
        running = False

        def __init__(self):
            self.listeners, self.started, self.stopped = [], False, False

        def on_tick(self, fn):
            self.listeners.append(fn)

        def off_tick(self, fn):
            self.listeners.remove(fn)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    fake = FakeStream()
    bot.stream = fake
    waits = []

    def fake_wait(until, ticks):  # instead of sleeping a day: feed a price, then let _wait drain it
        waits.append(until)
        nxt = bot._levels_for
        for fn in fake.listeners:
            fn(Tick("BTC/USDT", 96.0, pd.Timestamp.now(tz="UTC"), (nxt, 100, 100, 94.0, 96.0, 1), "websocket"))
        real_wait(time.time() + 0.05, ticks)

    real_wait = bot._wait
    monkeypatch.setattr(bot, "_wait", fake_wait)
    bot.run(max_steps=2)
    assert fake.started and fake.stopped and not fake.listeners and len(waits) == 1
    assert [e.action for e in bot.events][:2] == ["buy", "sell"] and "stop" in bot.events[1].note


def test_ticker_candles_keep_the_source_grid():
    """Yahoo's daily stock candles start at the exchange's midnight (05:00 UTC), not at 00:00 UTC."""
    s = MarketStream("AAPL", "1d", poll=lambda: None)
    day, five = 86_400_000, 5 * H
    s.candle = [five, 10.0, 11.0, 9.0, 10.0, 100.0]
    assert s._trade(12.0, five + 10 * H) and s.candle == [five, 10, 12, 9, 12, 100]
    assert s._trade(13.0, five + day + H) and s.candle == [five + day, 13, 13, 13, 13, 0]
    assert not s._trade(1.0, five)  # older than the forming candle


def test_bot_run_cli_attaches_a_stream(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from tradebot.cli import main as cli

    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "create_broker", lambda *a, **k: BinanceBroker(exchange=FakeExchange()))
    opened, ran = [], []

    class Fake:
        def on_state(self, fn):
            fn("polling", "no websocket")

    monkeypatch.setattr(st, "open_stream", lambda symbol, tf, broker=None: opened.append((symbol, tf)) or Fake())
    monkeypatch.setattr(TradingBot, "run", lambda self, max_steps=None: ran.append((self.stream, max_steps)))
    args = ["bot", "run", "BTC/USDT", "sma_cross", "--tf", "1h", "--no-journal"]
    res = CliRunner().invoke(cli.app, args)
    assert res.exit_code == 0, res.output
    assert opened == [("BTC/USDT", "1h")] and isinstance(ran[0][0], Fake) and "price stream: polling" in res.output
    for extra in (["--no-stream"], ["--once"]):
        ran.clear()
        assert CliRunner().invoke(cli.app, args + extra).exit_code == 0 and ran[0][0] is None
    assert len(opened) == 1
