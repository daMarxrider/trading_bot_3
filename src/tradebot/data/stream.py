# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Streaming prices over websockets, with reconnects and a polling fallback (#158).

A :class:`MarketStream` follows one symbol on a background thread and hands every price update to its
listeners as a :class:`Tick`: the last price and the candle that is forming (its open, high, low, close and
volume so far). Where it comes from:

* crypto: the exchange's websocket through ccxt pro (part of ccxt), ``watchOHLCV`` where the exchange has it,
  else ``watchTicker`` with the candle built from the prices on top of one REST snapshot;
* stocks and ETFs: Yahoo Finance's price stream (``yfinance.AsyncWebSocket``), during market hours;
* anything else, or when the websocket keeps failing: polling the REST candles every ``poll_interval``.

A dropped connection is reopened with a growing pause (``backoff``, doubling up to ``max_backoff``). After
``retries`` failed attempts in a row the stream polls instead and tries the websocket again after
``retry_after`` seconds. :attr:`MarketStream.state` says which it is doing (``connecting``, ``streaming``,
``reconnecting``, ``polling``, ``stopped``); state listeners hear every change.

Listeners run on the stream's thread: keep them short, hand the tick to your own thread (a queue, a Qt
signal). The live bot (:class:`tradebot.live.TradingBot`) and the chart's Live button use it.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from tradebot.core.models import AssetClass, guess_asset_class
from tradebot.core.timeframes import to_timedelta

log = logging.getLogger(__name__)

STATES = ("connecting", "streaming", "reconnecting", "polling", "stopped")


@dataclass(frozen=True)
class Tick:
    symbol: str
    price: float
    time: pd.Timestamp  #: when the update arrived (UTC)
    candle: tuple  #: the forming candle: (start Timestamp, open, high, low, close, volume)
    source: str  #: "websocket" or "polling"


def candle_start(ts_ms: float, timeframe: str) -> int:
    """Start (ms since the epoch) of the ``timeframe`` candle holding ``ts_ms``: weeks start on Monday,
    months on the first, everything else on a multiple of its length since 1970-01-01 UTC."""
    if timeframe in ("1w", "1mo"):
        day = pd.Timestamp(int(ts_ms), unit="ms", tz="UTC").normalize()
        day = day - pd.Timedelta(days=day.weekday()) if timeframe == "1w" else day.replace(day=1)
        return int(day.timestamp() * 1000)
    step = int(to_timedelta(timeframe).total_seconds() * 1000)
    return int(ts_ms) - int(ts_ms) % step


class MarketStream:
    """Live prices of ``symbol`` on ``timeframe`` candles.

    ``client_factory`` makes an async websocket client with ccxt pro's interface (``has``, ``watch_ohlcv`` or
    ``watch_ticker``, ``close``); ``poll`` returns a frame of the latest candles (the forming one last). Either
    may be None, not both. Use :func:`open_stream` to get both for a symbol."""

    def __init__(self, symbol: str, timeframe: str = "1m", client_factory: Callable | None = None,
                 poll: Callable[[], pd.DataFrame] | None = None, *, poll_interval: float = 5.0, retries: int = 3,
                 backoff: float = 1.0, max_backoff: float = 30.0, retry_after: float = 300.0):
        if client_factory is None and poll is None:
            raise ValueError("a stream needs a websocket client or a poll function")
        to_timedelta(timeframe)  # raises on an unknown timeframe
        self.symbol, self.timeframe = symbol, timeframe
        self.client_factory, self.poll = client_factory, poll
        self.poll_interval, self.retries, self.backoff = poll_interval, retries, backoff
        self.max_backoff, self.retry_after = max_backoff, retry_after
        self.state = "stopped"
        self.note = ""  #: why the stream is in its state (the last error), for a tooltip or the log
        self.candle: list | None = None  #: [start ms, open, high, low, close, volume] of the forming candle
        self.price: float | None = None
        self._ticks: list[Callable[[Tick], None]] = []
        self._states: list[Callable[[str, str], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._received = False

    # -- public
    def on_tick(self, fn: Callable[[Tick], None]) -> None:
        self._ticks.append(fn)

    def off_tick(self, fn: Callable[[Tick], None]) -> None:
        if fn in self._ticks:
            self._ticks.remove(fn)

    def on_state(self, fn: Callable[[str, str], None]) -> None:
        self._states.append(fn)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "MarketStream":
        if not self.running:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name=f"stream {self.symbol}", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        loop, task = self._loop, self._task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:  # the loop just closed
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        self._thread = None
        self._set_state("stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- the thread
    def _set_state(self, state: str, note: str = "") -> None:
        if state == self.state and note == self.note:
            return
        self.state, self.note = state, note
        log.info("stream %s %s: %s%s", self.symbol, self.timeframe, state, f" ({note})" if note else "")
        for fn in list(self._states):
            try:
                fn(state, note)
            except Exception:
                log.exception("stream state listener failed")

    def _emit(self, source: str) -> None:
        c = self.candle
        self.price = c[4]
        tick = Tick(self.symbol, float(c[4]), pd.Timestamp.now(tz="UTC"),
                    (pd.Timestamp(int(c[0]), unit="ms", tz="UTC"), *map(float, c[1:])), source)
        for fn in list(self._ticks):
            try:
                fn(tick)
            except Exception:
                log.exception("stream listener failed")

    def _row(self, row) -> bool:
        """Merge a candle ``[start ms, o, h, l, c, v]``; False for one older than the forming candle."""
        ts = int(row[0])
        if self.candle is not None and ts < self.candle[0]:
            return False
        self.candle = [ts, *(float(v or 0) for v in row[1:6])]
        while len(self.candle) < 6:
            self.candle.append(0.0)
        return True

    def _trade(self, price: float, ts_ms: float | None) -> bool:
        """Merge one price: a new candle when it falls into the next period. Candles go on from the source's own
        grid (Yahoo's daily stock candles start at the exchange's midnight, not at 00:00 UTC)."""
        ts = ts_ms if ts_ms else time.time() * 1000
        c = self.candle
        start = candle_start(ts, self.timeframe)
        if c is not None and self.timeframe not in ("1w", "1mo"):
            step = to_timedelta(self.timeframe).total_seconds() * 1000
            start = int(c[0] + (ts - c[0]) // step * step)
        if c is None or start > c[0]:
            self.candle = [start, price, price, price, price, 0.0]
        elif start == c[0]:
            c[2], c[3], c[4] = max(c[2], price), min(c[3], price), price
        else:
            return False
        return True

    def _poll_once(self) -> None:
        df = self.poll()
        if df is None or not len(df):
            return
        last = df.iloc[-1]
        row = [int(df.index[-1].timestamp() * 1000), last["open"], last["high"], last["low"], last["close"],
               last["volume"] if "volume" in df.columns else 0.0]
        if self._row(row):
            self._emit("polling")

    def _run(self) -> None:
        failures, ws_at = 0, 0.0
        while not self._stop.is_set():
            if self.client_factory is not None and time.monotonic() >= ws_at:
                self._set_state("connecting" if failures == 0 else "reconnecting", self.note if failures else "")
                self._received = False
                try:
                    asyncio.run(self._session())
                    error = "the stream ended"
                except asyncio.CancelledError:
                    error = "cancelled"
                except Exception as exc:  # network, exchange or parsing errors: reconnect
                    error = f"{type(exc).__name__}: {exc}"
                if self._stop.is_set():
                    break
                failures = 1 if self._received else failures + 1  # a session that delivered data starts over
                log.warning("stream %s: %s", self.symbol, error)
                if failures >= self.retries:
                    failures, ws_at = 0, time.monotonic() + self.retry_after
                    if self.poll is not None:
                        self._set_state("polling", f"websocket failed: {error}")
                    else:
                        self._set_state("reconnecting", f"websocket failed: {error}")
                        self._stop.wait(min(self.retry_after, self.max_backoff))
                    continue
                self._set_state("reconnecting", error)
                self._stop.wait(min(self.backoff * 2 ** (failures - 1), self.max_backoff))
            elif self.poll is not None:
                if self.client_factory is None:
                    self._set_state("polling", "no websocket for this symbol")
                try:
                    self._poll_once()
                except Exception as exc:
                    log.warning("stream %s: poll failed: %s", self.symbol, exc)
                    self.note = f"poll failed: {exc}"
                self._stop.wait(self.poll_interval)
            else:
                self._stop.wait(max(0.0, min(ws_at - time.monotonic(), self.max_backoff)))
        self._loop = self._task = None

    def _got(self) -> None:
        if not self._received:
            self._received = True
            self._set_state("streaming")
        self._emit("websocket")

    async def _session(self) -> None:
        self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
        if self._stop.is_set():
            return
        client = self.client_factory()
        try:
            has = getattr(client, "has", None) or {}
            ohlcv = bool(has.get("watchOHLCV")) and hasattr(client, "watch_ohlcv")
            if not ohlcv and self.candle is None and self.poll is not None:
                try:  # the candle so far, so prices from the stream extend its real open, high and low
                    self._poll_once()
                except Exception as exc:
                    log.debug("stream %s: no snapshot: %s", self.symbol, exc)
            while not self._stop.is_set():
                if ohlcv:  # one tick per row, so a candle that just closed gets its last values out too
                    rows = await client.watch_ohlcv(self.symbol, self.timeframe)
                    for r in sorted(rows or [], key=lambda r: r[0]):
                        if self._row(r):
                            self._got()
                else:
                    t = await client.watch_ticker(self.symbol) or {}
                    price = t.get("last") or t.get("close")
                    if price is not None and self._trade(float(price), t.get("timestamp")):
                        self._got()
        finally:
            try:
                closed = client.close()
                if inspect.isawaitable(closed):
                    await closed
            except Exception:
                pass


class YahooStreamClient:
    """Yahoo Finance's price stream (stocks, ETFs) with the ``watch_ticker`` interface of a ccxt pro client.

    Speaks the protocol of ``yfinance.AsyncWebSocket`` (JSON ``subscribe`` messages, prices as base64
    protobuf, the subscription repeated every 15 s as a heartbeat) over ``websockets`` itself, as yfinance's
    ``listen`` loops forever on a closed connection instead of ending, which a reconnect needs."""

    URL = "wss://streamer.finance.yahoo.com/?version=2"
    HEARTBEAT = 15.0
    has = {"watchTicker": True}

    def __init__(self, url: str | None = None):
        self.url = url or self.URL
        self._ws = self._beat = None
        self._wanted = ""

    async def _heartbeat(self, message: str) -> None:
        while True:
            await asyncio.sleep(self.HEARTBEAT)
            await self._ws.send(message)

    async def watch_ticker(self, symbol: str) -> dict:
        from google.protobuf.json_format import MessageToDict
        from websockets.asyncio.client import connect
        from yfinance.pricing_pb2 import PricingData

        from tradebot.data.providers import YFinanceProvider

        if self._ws is None:
            self._wanted = YFinanceProvider.map_symbol(symbol)
            self._ws = await connect(self.url)
            message = json.dumps({"subscribe": [self._wanted]})
            await self._ws.send(message)
            self._beat = asyncio.ensure_future(self._heartbeat(message))
        while True:
            raw = await self._ws.recv()  # raises ConnectionClosed when the server hangs up
            data = PricingData()
            data.ParseFromString(base64.b64decode(json.loads(raw).get("message", "")))
            msg = MessageToDict(data, preserving_proto_field_name=True)
            if msg.get("price") and msg.get("id", self._wanted) == self._wanted:
                stamp = msg.get("time")
                return {"last": float(msg["price"]), "timestamp": int(stamp) if stamp else None}

    async def close(self) -> None:
        if self._beat is not None:
            self._beat.cancel()
        if self._ws is not None:
            await self._ws.close()


def ccxt_pro_factory(exchange_id: str, sandbox: bool = False) -> Callable | None:
    """A factory for ccxt pro's client of ``exchange_id``, or None when it has no candle or price stream."""
    try:
        import ccxt.pro as ccxtpro
    except Exception:  # pragma: no cover - an old ccxt
        return None
    cls = getattr(ccxtpro, exchange_id, None)
    if cls is None:
        return None
    try:
        has = cls().has  # the class attribute lacks the exchange's own entries; an instance opens no connection
    except Exception:
        return None
    if not (has.get("watchOHLCV") or has.get("watchTicker")):
        return None

    def make():
        client = cls({"enableRateLimit": True})
        if sandbox:
            client.set_sandbox_mode(True)
        return client

    return make


def provider_poller(engine, symbol: str, timeframe: str) -> Callable[[], pd.DataFrame] | None:
    """Recent candles straight from the first of ``engine``'s providers that serves ``symbol`` (not through the
    cache, which doesn't refetch a series it fetched a minute ago)."""
    asset = guess_asset_class(symbol)
    providers = [p for p in getattr(engine, "providers", []) if p.supports(asset, timeframe)]
    if not providers:
        return None
    step = to_timedelta(timeframe)

    def poll():
        end = pd.Timestamp.now(tz="UTC") + step
        error = None
        for p in providers:
            try:
                return p.fetch(symbol, timeframe, end - 4 * step, end)
            except Exception as exc:
                error = exc
        raise error

    return poll


def open_stream(symbol: str, timeframe: str = "1m", *, broker=None, engine=None, exchange_id: str | None = None,
                websocket: bool = True, **kwargs) -> MarketStream:
    """A :class:`MarketStream` for ``symbol``: the websocket that fits it, polling as the fallback.

    Crypto streams from ``exchange_id``, else the broker's exchange (a ccxt broker), else the first exchange
    of the data engine (Binance by default). Polling goes through the broker's candles for a ccxt broker and
    the data providers otherwise. ``websocket=False`` only polls. Not started yet: call ``start()``."""
    factory = None
    if websocket:
        if guess_asset_class(symbol) is AssetClass.CRYPTO:
            ex = exchange_id or getattr(broker, "exchange_id", None)
            data = engine if engine is not None else getattr(broker, "data", None)  # the paper broker: its data
            if ex is None and data is not None:
                ex = next((p.name.split(":", 1)[1] for p in getattr(data, "providers", [])
                           if p.name.startswith("ccxt:")), None)
            sandbox = bool(getattr(getattr(broker, "exchange", None), "isSandboxModeEnabled", False))
            factory = ccxt_pro_factory(ex or "binance", sandbox)
        else:
            factory = YahooStreamClient
    poll = None
    if broker is not None and hasattr(broker, "exchange_id"):
        poll = lambda: broker.get_klines(symbol, timeframe, 3)  # noqa: E731
    else:
        poll = provider_poller(engine if engine is not None else getattr(broker, "data", None), symbol, timeframe)
    if factory is None and poll is None:
        raise ValueError(f"no stream and no data provider for {symbol} {timeframe}")
    return MarketStream(symbol, timeframe, factory, poll, **kwargs)
