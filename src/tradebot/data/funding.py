# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Historical funding rates of crypto perpetual futures (#149), for the backtest's margin model.

A perpetual settles funding every few hours (8 on Binance): a positive rate means longs pay shorts that share of
their notional. :func:`fetch_history` downloads the settled rates from a ccxt exchange with perpetuals (Binance
USD-M futures by default; ``BTC/USDT`` is looked up as its perpetual ``BTC/USDT:USDT``) and :func:`per_candle`
sums them per candle of a chart, so the backtester charges each candle the funding settled while it was held.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_EXCHANGE = "binanceusdm"


def perpetual(symbol: str) -> str:
    """The ccxt perpetual symbol of a spot pair: ``BTC/USDT`` -> ``BTC/USDT:USDT`` (unchanged when it has one)."""
    if ":" in symbol or "/" not in symbol:
        return symbol
    return f"{symbol}:{symbol.split('/')[1]}"


def fetch_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp, exchange=None,
                  exchange_id: str = DEFAULT_EXCHANGE, page_limit: int = 1000) -> pd.Series:
    """Settled funding rates of ``symbol``'s perpetual from ``start`` to ``end`` (a series of fractions indexed
    by the settlement time, UTC). Raises ``RuntimeError`` when the exchange has no funding history for it."""
    if exchange is None:
        import ccxt

        exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    if not exchange.has.get("fetchFundingRateHistory"):
        raise RuntimeError(f"{exchange_id} has no funding rate history")
    market = perpetual(symbol)
    since, end_ms = int(pd.Timestamp(start).timestamp() * 1000), int(pd.Timestamp(end).timestamp() * 1000)
    rows: dict[int, float] = {}
    while since < end_ms:
        page = exchange.fetch_funding_rate_history(market, since=since, limit=page_limit)
        if not page:
            break
        rows.update((int(r["timestamp"]), float(r["fundingRate"])) for r in page if r["timestamp"] <= end_ms)
        nxt = int(page[-1]["timestamp"]) + 1
        if nxt <= since:
            break
        since = nxt
    if not rows:
        raise RuntimeError(f"no funding history for {market}")
    ts = sorted(rows)
    return pd.Series([rows[t] for t in ts], index=pd.to_datetime(ts, unit="ms", utc=True), name="funding")


def per_candle(rates: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Funding charged per candle of ``index`` (candle open times): candle ``t`` is held from the close of ``t-1``
    (≈ its open) to its own close (≈ the next open), so it pays the settlements in ``[index[t], index[t+1])``;
    the last candle pays those in the typical candle length after its open. 0 where nothing was settled."""
    if len(index) == 0:
        return np.zeros(0)
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    t = rates.index
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    cum = np.r_[0.0, np.cumsum(rates.to_numpy(dtype=float))]
    step = idx[1:] - idx[:-1]
    last = idx[-1] + (pd.Series(step).median() if len(step) else pd.Timedelta(days=1))
    edges = idx.append(pd.DatetimeIndex([last]))
    pos = np.searchsorted(t.as_unit("ns").asi8, edges.as_unit("ns").asi8, side="left")
    return np.diff(cum[pos])
