# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Stock & crypto screener.

Builds one row of technical/fundamental columns per symbol, then filters with
plain expressions, e.g. ``["rsi < 35", "close > sma200", "market_cap > 1e10"]``.
Extra columns can be any registered indicator output: ``extra={"stoch_k": ("stoch", {"period": 14}, "k")}``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import pandas as pd

from tradebot.core.series import OHLC
from tradebot.indicators import INDICATORS
from tradebot.indicators.builtin import ATR, RSI, SMA

log = logging.getLogger(__name__)

COLUMNS = ["symbol", "close", "change_1d", "change_1m", "change_1y", "rsi", "sma50", "sma200",
           "atr_pct", "avg_volume", "market_cap"]


def screen_row(symbol: str, df: pd.DataFrame, market_cap: float | None = None,
               extra: dict[str, tuple[str, dict, str | None]] | None = None) -> dict:
    o = OHLC.from_frame(df)
    c = df["close"]

    def change(n):
        return float(c.iloc[-1] / c.iloc[-1 - n] - 1) if len(c) > n else np.nan

    row = {
        "symbol": symbol,
        "close": float(c.iloc[-1]),
        "change_1d": change(1),
        "change_1m": change(21),
        "change_1y": change(252),
        "rsi": float(RSI.compute(o, period=14)["value"][-1]),
        "sma50": float(SMA.compute(o, period=50)["value"][-1]),
        "sma200": float(SMA.compute(o, period=200)["value"][-1]),
        "atr_pct": float(ATR.compute(o, period=14)["value"][-1] / c.iloc[-1]),
        "avg_volume": float(df["volume"].tail(20).mean()) if "volume" in df else np.nan,
        "market_cap": market_cap if market_cap is not None else np.nan,
    }
    for col, (name, params, output) in (extra or {}).items():
        ind = INDICATORS.get(name)(**params)
        out = ind.compute(o, **ind.values)
        row[col] = float(np.asarray(out[output or ind.outputs[0]])[-1])
    return row


def yf_market_cap(symbol: str) -> float | None:
    if "/" in symbol:
        return None
    try:
        import yfinance as yf

        return float(yf.Ticker(symbol).fast_info["marketCap"])
    except Exception:
        return None


def run_screen(
    symbols: list[str],
    fetch: Callable[[str], pd.DataFrame],
    filters: list[str] | None = None,
    extra: dict | None = None,
    market_cap: Callable[[str], float | None] = yf_market_cap,
    workers: int = 8,
) -> pd.DataFrame:
    """Evaluate ``symbols`` concurrently; rows failing a filter (or lacking data) are dropped."""

    def one(sym):
        try:
            return screen_row(sym, fetch(sym), market_cap(sym), extra)
        except Exception as exc:
            log.warning("screener skipped %s: %s", sym, exc)
            return None

    with ThreadPoolExecutor(workers) as pool:
        rows = [r for r in pool.map(one, symbols) if r]
    table = pd.DataFrame(rows, columns=COLUMNS + list(extra or {}))
    for expr in filters or []:
        table = table.query(expr, engine="python")
    return table.reset_index(drop=True)
