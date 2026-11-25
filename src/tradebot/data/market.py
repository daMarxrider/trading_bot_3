# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Crypto market data beyond candles (#166): funding rates, open interest, the fear and greed index, on-chain
metrics and order-book depth, all from free endpoints that need no key.

========================  =====================================================  ===========================
frame column              source                                                 known from
========================  =====================================================  ===========================
``funding_rate``          the perpetual's settled rates (ccxt, Binance USD-M)    its settlement time
``open_interest``         open interest history in quote currency (ccxt)         the end of its bucket
``fear_greed``            alternative.me Fear & Greed index (0-100, daily)       its day's 00:00 UTC
``mvrv``                  Coin Metrics community API, ``CapMVRVCur`` (daily)     the end of its day
``active_addresses``      Coin Metrics community API, ``AdrActCnt`` (daily)      the end of its day
========================  =====================================================  ===========================

Exchange flows and most other on-chain metrics have no free keyless source and are not offered.

:meth:`MarketData.attach` adds the columns to a candle frame. Every value is lined up **without look-ahead**: a
candle sees the last value that was *known at its open time* (see the last column; :func:`align`), so a strategy on
bar ``t`` never reads something published during or after it. A value older than the series' ``max_age`` is NaN,
instead of being carried forward for ever. Indicators (:mod:`tradebot.indicators.market`) read the columns.

Everything is cached on disk next to the candles (``market/<column>/<symbol>.npz``), so history accumulates beyond
what the source serves (Binance keeps 30 days of open interest) and a chart works offline. A source that fails
(offline, rate limited, no perpetual for the symbol) leaves its column out and says why in
``df.attrs["market_errors"]``; nothing raises.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.core.models import AssetClass, guess_asset_class, split_pair
from tradebot.data import funding as funding_data

log = logging.getLogger(__name__)

USER_AGENT = "tradebot (open-source trading app)"
TIMEOUT = 15.0
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0&format=json"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
DEFAULT_EXCHANGE = "binanceusdm"  #: perpetuals: funding rate and open interest
BOOK_EXCHANGE = "binance"  #: spot: the depth chart
FRESH_FOR = 300.0  #: seconds a cached series counts as just fetched
HOUR, DAY = pd.Timedelta(hours=1), pd.Timedelta(days=1)
#: the open interest buckets exchanges serve, ccxt timeframes
OI_TIMEFRAMES = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
                 "1h": HOUR, "2h": 2 * HOUR, "4h": 4 * HOUR, "6h": 6 * HOUR, "12h": 12 * HOUR, "1d": DAY}
#: how far back an exchange serves open interest history (Binance: 30 days)
OI_HISTORY = {"binanceusdm": pd.Timedelta(days=29)}


class MarketDataError(RuntimeError):
    """A market data source can't deliver (offline, rate limited, unknown symbol, not a crypto pair)."""


@dataclass(frozen=True)
class Source:
    """One market data column: what it is, how stale it may get and when a value becomes known."""

    column: str
    title: str
    max_age: pd.Timedelta  #: a value older than this at a candle is NaN
    lag: pd.Timedelta  #: a value stamped ``t`` is known from ``t + lag``
    daily: bool = False
    needs_perpetual: bool = False


SOURCES: dict[str, Source] = {s.column: s for s in (
    Source("funding_rate", "Funding rate", 2 * DAY, pd.Timedelta(0), needs_perpetual=True),
    Source("open_interest", "Open interest", 2 * DAY, pd.Timedelta(0), needs_perpetual=True),  # lag = its bucket
    Source("fear_greed", "Fear & Greed index", 3 * DAY, pd.Timedelta(0), daily=True),
    Source("mvrv", "MVRV", 4 * DAY, DAY, daily=True),
    Source("active_addresses", "Active addresses", 4 * DAY, DAY, daily=True),
    Source("exchange_netflow", "Exchange net flow", 4 * DAY, DAY, daily=True),
)}
#: column -> Coin Metrics metrics; the value is the first minus the others (net flow = inflow - outflow)
COINMETRICS = {"mvrv": ("CapMVRVCur",), "active_addresses": ("AdrActCnt",),
               "exchange_netflow": ("FlowInExUSD", "FlowOutExUSD")}


def is_crypto(symbol: str) -> bool:
    """A crypto pair (``BTC/USDT``, or its perpetual ``BTC/USDT:USDT``), not a stock or a spread of two symbols."""
    return guess_asset_class(symbol) == AssetClass.CRYPTO and split_pair(symbol) is None


def base_asset(symbol: str) -> str:
    """``BTC/USDT`` -> ``btc`` (Coin Metrics' asset id)."""
    return symbol.split("/")[0].lower()


def http_json(url: str):
    """GET ``url`` as JSON (standard library only); raises on network or HTTP errors."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 - fixed https endpoints
        return json.loads(resp.read().decode("utf-8"))


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _series(times_ms, values, name: str) -> pd.Series:
    s = pd.Series(np.asarray(values, dtype=float), index=pd.to_datetime(np.asarray(times_ms, dtype="int64"),
                                                                        unit="ms", utc=True), name=name)
    return s[~s.index.duplicated(keep="last")].sort_index()


# -- fetching ---------------------------------------------------------------------------------------
def oi_timeframe(step: pd.Timedelta) -> str:
    """The open interest bucket for candles of length ``step``: the longest one that is not longer than a candle."""
    best = "5m"
    for tf, length in OI_TIMEFRAMES.items():
        if length <= step:
            best = tf
    return best


def _exchange(exchange_id: str):
    import ccxt

    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def fetch_open_interest(symbol: str, timeframe: str, start, end, exchange=None, exchange_id: str = DEFAULT_EXCHANGE,
                        page_limit: int = 500) -> pd.Series:
    """Open interest of ``symbol``'s perpetual in quote currency (else contracts), per ``timeframe`` bucket from
    ``start`` to ``end``, indexed by the bucket's timestamp (UTC)."""
    if exchange is None:
        exchange = _exchange(exchange_id)
    if not exchange.has.get("fetchOpenInterestHistory"):
        raise MarketDataError(f"{exchange_id} has no open interest history")
    market = funding_data.perpetual(symbol)
    reach = OI_HISTORY.get(exchange_id)
    start, end = _utc(start), _utc(end)
    if reach is not None:
        start = max(start, pd.Timestamp.now(tz="UTC") - reach)
    since, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    rows: dict[int, float] = {}
    while since < end_ms:
        page = exchange.fetch_open_interest_history(market, timeframe, since=since, limit=page_limit,
                                                    params={"until": end_ms})
        if not page:
            break
        for r in page:
            value = r.get("openInterestValue")
            value = r.get("openInterestAmount") if value is None else value
            if value is not None and r["timestamp"] <= end_ms:
                rows[int(r["timestamp"])] = float(value)
        nxt = int(page[-1]["timestamp"]) + 1
        if nxt <= since:
            break
        since = nxt
    if not rows:
        raise MarketDataError(f"no open interest history for {market}")
    ts = sorted(rows)
    return _series(ts, [rows[t] for t in ts], "open_interest")


def fetch_fear_greed(get_json=http_json) -> pd.Series:
    """The whole daily Fear & Greed history (0 = extreme fear, 100 = extreme greed), stamped 00:00 UTC of its day."""
    data = get_json(FEAR_GREED_URL).get("data") or []
    if not data:
        raise MarketDataError("the Fear & Greed index returned no data")
    return _series([int(d["timestamp"]) * 1000 for d in data], [float(d["value"]) for d in data], "fear_greed")


def fetch_onchain(column: str, symbol: str, start, end, get_json=http_json) -> pd.Series:
    """A daily on-chain metric of ``symbol``'s base asset from the Coin Metrics community API (stamped 00:00 UTC of
    its day; known only once the day is over)."""
    metrics = COINMETRICS[column]
    url = (f"{COINMETRICS_URL}?assets={base_asset(symbol)}&metrics={','.join(metrics)}&frequency=1d&page_size=10000"
           f"&start_time={_utc(start):%Y-%m-%d}&end_time={_utc(end):%Y-%m-%d}")
    times, values = [], []
    for _ in range(50):  # pages
        body = get_json(url)
        if body.get("error"):
            raise MarketDataError(f"Coin Metrics: {body['error'].get('message', body['error'])}")
        for row in body.get("data", []):
            if all(row.get(m) is not None for m in metrics):
                times.append(int(pd.Timestamp(row["time"]).timestamp() * 1000))
                first, *rest = (float(row[m]) for m in metrics)
                values.append(first - sum(rest))
        url = body.get("next_page_url")
        if not url:
            break
    if not values:
        raise MarketDataError(f"no {column} data for {base_asset(symbol).upper()} (Coin Metrics covers the major coins)")
    return _series(times, values, column)


# -- lining up --------------------------------------------------------------------------------------
def align(values: pd.Series, index: pd.DatetimeIndex, lag: pd.Timedelta = pd.Timedelta(0),
          max_age: pd.Timedelta | None = None) -> np.ndarray:
    """``values`` per candle of ``index`` (candle open times) **without look-ahead**: candle ``t`` gets the last
    value known at ``t``, i.e. stamped at or before ``t - lag``, and NaN when that is older than ``max_age`` or
    there is none yet."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    if values.empty or len(idx) == 0:
        return np.full(len(idx), np.nan)
    known = (values.index.tz_localize("UTC") if values.index.tz is None else values.index.tz_convert("UTC")) + lag
    known_ns, v = known.as_unit("ns").asi8, values.to_numpy(dtype=float)
    order = np.argsort(known_ns, kind="stable")
    known_ns, v = known_ns[order], v[order]
    at = np.searchsorted(known_ns, idx.as_unit("ns").asi8, side="right") - 1
    out = np.where(at >= 0, v[np.maximum(at, 0)], np.nan)
    if max_age is not None:
        age = idx.as_unit("ns").asi8 - known_ns[np.maximum(at, 0)]
        out = np.where(age > max_age.value, np.nan, out)
    return out


# -- disk cache -------------------------------------------------------------------------------------
class SeriesCache:
    """``market/<column>/<symbol>.npz`` in the candle cache's folder: one timestamp/value series each."""

    def __init__(self, folder: Path | None):
        self.folder = Path(folder) / "market" if folder is not None else None

    def path(self, column: str, symbol: str, variant: str = "") -> Path:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", symbol) + (f"@{variant}" if variant else "")
        return self.folder / column / f"{name}.npz"

    def load(self, column, symbol, variant="") -> tuple[pd.Series, float, pd.Timestamp] | None:
        """``(series, when it was fetched, from when on it was asked for)`` or None."""
        if self.folder is None:
            return None
        path = self.path(column, symbol, variant)
        try:
            with np.load(path, allow_pickle=False) as data:
                s = _series(data["t"], data["v"], column)
                covered = pd.Timestamp(int(data["covered"]), unit="ms", tz="UTC")
            return s, path.stat().st_mtime, covered
        except Exception:  # noqa: BLE001 - missing, corrupt or from another version
            return None

    def save(self, column, symbol, series: pd.Series, covered: pd.Timestamp, variant="") -> None:
        if self.folder is None or series.empty:
            return
        path = self.path(column, symbol, variant)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("wb") as fh:
                np.savez(fh, t=series.index.as_unit("ms").asi8, v=series.to_numpy(dtype=float),
                         covered=np.int64(round(covered.timestamp() * 1000)))
            tmp.replace(path)
        except OSError as exc:
            log.info("could not write market data cache %s: %s", path, exc)


# -- order book -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderBook:
    """A snapshot of an order book: ``bids`` (best first, falling) and ``asks`` (best first, rising) as
    ``(price, amount)`` rows."""

    symbol: str
    exchange: str
    time: pd.Timestamp
    bids: np.ndarray = field(repr=False)
    asks: np.ndarray = field(repr=False)

    @property
    def mid(self) -> float:
        return float((self.bids[0, 0] + self.asks[0, 0]) / 2)

    @property
    def spread(self) -> float:
        """The best ask over the best bid, as a share of the mid price."""
        return float((self.asks[0, 0] - self.bids[0, 0]) / self.mid)

    def depth(self, within: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(bid_price, bid_cum, ask_price, ask_cum)``: the cumulative amount that can be traded down to / up to
        each price, optionally only ``within`` (0.05 = 5 %) of the mid price. Bid prices run from the lowest
        up to the best bid, so both sides are ready to plot as steps."""
        mid = self.mid
        bids = self.bids if within is None else self.bids[self.bids[:, 0] >= mid * (1 - within)]
        asks = self.asks if within is None else self.asks[self.asks[:, 0] <= mid * (1 + within)]
        return bids[::-1, 0], np.cumsum(bids[:, 1])[::-1], asks[:, 0], np.cumsum(asks[:, 1])

    def imbalance(self, within: float = 0.02) -> float:
        """Bid amount minus ask amount over their sum within ``within`` of the mid price: +1 all bids, -1 all asks."""
        _, bc, _, ac = self.depth(within)
        b, a = (bc[0] if len(bc) else 0.0), (ac[-1] if len(ac) else 0.0)
        return float((b - a) / (b + a)) if b + a > 0 else 0.0


def fetch_order_book(symbol: str, limit: int = 500, exchange=None, exchange_id: str = BOOK_EXCHANGE) -> OrderBook:
    """The current order book of ``symbol`` (``limit`` levels a side). Raises :class:`MarketDataError`."""
    try:
        if exchange is None:
            exchange = _exchange(exchange_id)
        raw = exchange.fetch_order_book(symbol, limit)
    except Exception as exc:  # noqa: BLE001 - offline, rate limit, unknown market
        raise MarketDataError(f"order book of {symbol}: {exc}") from exc
    bids = np.array([[r[0], r[1]] for r in raw.get("bids", [])], dtype=float).reshape(-1, 2)
    asks = np.array([[r[0], r[1]] for r in raw.get("asks", [])], dtype=float).reshape(-1, 2)
    if not len(bids) or not len(asks):
        raise MarketDataError(f"{symbol}: the order book is empty")
    stamp = pd.Timestamp(raw["timestamp"], unit="ms", tz="UTC") if raw.get("timestamp") else pd.Timestamp.now(tz="UTC")
    return OrderBook(symbol, getattr(exchange, "id", exchange_id), stamp, bids, asks)


# -- the facade -------------------------------------------------------------------------------------
class MarketData:
    """Fetches, caches and lines up market data. Every network piece can be replaced (tests use fakes):
    ``exchange`` (a ccxt-like object for perpetuals), ``spot`` (the same for order books), ``get_json`` (for the
    HTTP sources). ``cache``: True = the candle cache's folder, a path, or None for no disk cache."""

    def __init__(self, cache: bool | Path | None = True, exchange=None, exchange_id: str = DEFAULT_EXCHANGE,
                 get_json=http_json, spot=None):
        folder = cache if isinstance(cache, Path) else None
        if cache is True:
            from tradebot.data.cache import DiskCache

            disk = DiskCache.from_config()
            folder = disk.folder if disk is not None else None
        self.cache = SeriesCache(folder)
        self.exchange, self.exchange_id, self.get_json, self.spot = exchange, exchange_id, get_json, spot

    def _perp(self):
        if self.exchange is None:
            self.exchange = _exchange(self.exchange_id)
        return self.exchange

    def _download(self, column: str, symbol: str, start, end, step: pd.Timedelta) -> tuple[pd.Series, str]:
        """``(series, variant)`` from the source; any failure becomes a :class:`MarketDataError`."""
        try:
            if column == "funding_rate":
                return funding_data.fetch_history(symbol, start, end, exchange=self._perp(),
                                                  exchange_id=self.exchange_id), ""
            if column == "open_interest":
                tf = oi_timeframe(step)
                return fetch_open_interest(symbol, tf, start, end, exchange=self._perp(),
                                           exchange_id=self.exchange_id), tf
            if column == "fear_greed":
                return fetch_fear_greed(self.get_json), ""
            return fetch_onchain(column, symbol, start, end, self.get_json), ""
        except MarketDataError:
            raise
        except Exception as exc:  # noqa: BLE001 - offline, rate limited, unknown market
            raise MarketDataError(str(exc)) from exc

    def series(self, column: str, symbol: str, start, end, step: pd.Timedelta = DAY) -> tuple[pd.Series, str, bool]:
        """``(series, variant, offline)`` of ``column`` for ``symbol`` over ``[start, end)``: the cache extended by
        what the source serves. A failing source falls back on the cache (``offline`` True); without one it
        raises :class:`MarketDataError`. ``variant`` is the open interest bucket."""
        src = SOURCES[column]
        start, end = _utc(start) - 2 * src.max_age, _utc(end)
        variant = oi_timeframe(step) if column == "open_interest" else ""
        # market-wide indices are one series for every symbol
        key = "all" if column == "fear_greed" else symbol
        cached = self.cache.load(column, key, variant)
        old, fetched_at, covered = cached if cached is not None else (None, 0.0, start)
        if old is not None and time.time() - fetched_at < FRESH_FOR:
            return old, variant, False
        fetch_from = start
        if old is not None and covered <= start:  # the cache was asked for this far back: only the tail
            fetch_from = max(start, old.index[-1] - max(step, DAY))
        try:
            new, variant = self._download(column, symbol, fetch_from, end, step)
        except MarketDataError as exc:
            if old is None:
                raise
            log.info("%s of %s: using the cache (%s)", column, symbol, exc)
            return old, variant, True
        merged = new if old is None else pd.concat([old[old.index < new.index[0]], new])
        self.cache.save(column, key, merged, min(covered, fetch_from), variant)
        return merged, variant, False

    def attach(self, df: pd.DataFrame, symbol: str, columns=None) -> pd.DataFrame:
        """``df`` with the market data ``columns`` (default: all that apply to ``symbol``) added, lined up with its
        candles without look-ahead (:func:`align`). A column that can't be had is left out and named in
        ``df.attrs["market_errors"]`` (``{column: reason}``); ``attrs["offline"]`` is set when the cache stood in."""
        out = df.copy()
        errors: dict[str, str] = {}
        offline = bool(df.attrs.get("offline"))
        if not is_crypto(symbol):
            out.attrs["market_errors"] = {c: f"{symbol} is not a crypto pair" for c in (columns or SOURCES)}
            return out
        cols = list(columns) if columns is not None else list(SOURCES)
        if len(df) == 0:
            return out
        step = _median_step(df.index)
        for column in cols:
            src = SOURCES[column]
            try:
                values, variant, off = self.series(column, symbol, df.index[0], df.index[-1] + step, step)
            except MarketDataError as exc:
                errors[column] = str(exc)
                continue
            offline = offline or off
            lag = OI_TIMEFRAMES[variant] if variant else src.lag  # an open interest bucket is known when it ends
            out[column] = align(values, df.index, lag, src.max_age)
        out.attrs["market_errors"] = errors
        out.attrs["market_data"] = [c for c in cols if c in out.columns]
        if offline:
            out.attrs["offline"] = True
        return out

    def order_book(self, symbol: str, limit: int = 500) -> OrderBook:
        return fetch_order_book(symbol, limit, exchange=self.spot, exchange_id=BOOK_EXCHANGE)


def _median_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return DAY
    return pd.Series(index[1:] - index[:-1]).median()


def needed(text: str) -> list[str]:
    """The market data columns the indicators named in ``text`` (a rule, an algorithm's parameters) need."""
    from tradebot.indicators import INDICATORS

    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text or ""))
    cols: list[str] = []
    for cls in INDICATORS:
        if cls.name in words:
            cols += [c for c in getattr(cls, "market_data", ()) if c not in cols]
    return cols


def wanted_by(indicators) -> list[str]:
    """The market data columns of indicator instances or classes."""
    cols: list[str] = []
    for ind in indicators:
        cols += [c for c in getattr(ind, "market_data", ()) if c not in cols]
    return cols


def missing(df: pd.DataFrame, columns) -> list[str]:
    """Which of ``columns`` ``df`` doesn't have yet."""
    return [c for c in columns if c not in df.columns]
