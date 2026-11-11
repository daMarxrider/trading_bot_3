# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Scanner: one indicator over a symbol universe, filtered on its plot values and alert states (#109).

Next to :mod:`tradebot.screener`, which stays as it is. The indicator only ever sees the last
:data:`BARS` candles of each symbol, so a result doesn't depend on what the cache holds::

    result = scan(["AAPL", "MSFT"], "ma_cross", {"fast": 9}, "1h",
                  filters=[{"alert": "cross_up", "state": True}, {"output": "fast", "op": ">", "value": "slow"}])
    result.table, result.summary()

A row has the symbol and the last-bar value of every output of the indicator (alerts as ``True``/``False``).
"""

from __future__ import annotations

import logging
import operator
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from tradebot.core.registry import resolve_params
from tradebot.core.series import OHLC
from tradebot.indicators import INDICATORS
from tradebot.indicators.base import is_screenable

log = logging.getLogger(__name__)

BARS = 500
MAX_SYMBOLS = 4000
TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1mo")
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne}

_lock = threading.Lock()


class ScanBusy(RuntimeError):
    """A scan is already running in this process."""


@dataclass
class ScanResult:
    table: pd.DataFrame  #: the matching rows only
    scanned: int  #: symbols that were computed (the filters were applied to them)
    matched: int
    skipped: dict[str, str] = field(default_factory=dict)  #: ``{symbol: reason}``, e.g. no data
    seconds: float = 0.0
    cancelled: bool = False

    def summary(self) -> str:
        text = f"scanned {self.scanned} · {self.matched} match · {len(self.skipped)} skipped · {self.seconds:.1f} s"
        return text + (" · cancelled" if self.cancelled else "")


def default_fetch(symbol: str, timeframe: str) -> pd.DataFrame:
    return _engine().get_ohlc(symbol, timeframe, bars=BARS)


_shared = None


def _engine():
    global _shared
    if _shared is None:
        from tradebot.data import MarketDataEngine

        _shared = MarketDataEngine()
    return _shared


def check_filters(cls, filters: list[dict]) -> list[dict]:
    """Validate ``filters`` against the indicator's own outputs; returns them normalised."""
    out = []
    for f in filters:
        if "alert" in f:
            if f["alert"] not in cls.alerts:
                raise ValueError(f"{cls.name} has no alert {f['alert']!r}; alerts: {', '.join(cls.alerts) or 'none'}")
            out.append({"alert": f["alert"], "state": bool(f.get("state", True))})
            continue
        if "output" not in f:
            raise ValueError(f"filter needs 'output' or 'alert': {f!r}")
        op, value = f.get("op"), f.get("value")
        if f["output"] not in cls.outputs:
            raise ValueError(f"{cls.name} has no output {f['output']!r}; outputs: {', '.join(cls.outputs)}")
        if op not in OPS:
            raise ValueError(f"unknown operator {op!r}; use one of {' '.join(OPS)}")
        if isinstance(value, str):
            if value not in cls.outputs:
                raise ValueError(f"{cls.name} has no output {value!r}; outputs: {', '.join(cls.outputs)}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"filter value must be a number or an output name: {value!r}")
        out.append({"output": f["output"], "op": op, "value": value})
    return out


def _passes(row: dict, filters: list[dict]) -> bool:
    for f in filters:
        if "alert" in f:
            if row[f["alert"]] != f["state"]:
                return False
            continue
        left = row[f["output"]]
        right = row[f["value"]] if isinstance(f["value"], str) else f["value"]
        if isinstance(left, bool) or isinstance(right, bool):
            left, right = float(left), float(right)
        if np.isnan(left) or np.isnan(right) or not OPS[f["op"]](left, right):  # warm-up never matches
            return False
    return True


def scan(
    symbols: list[str],
    indicator: str = "ma_cross",
    params: dict | None = None,
    timeframe: str = "1d",
    filters: list[dict] | None = None,
    fetch: Callable[[str, str], pd.DataFrame] | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    workers: int = 8,
) -> ScanResult:
    """Run ``indicator`` over ``symbols`` and keep the rows passing every filter (ANDed).

    ``fetch(symbol, timeframe)`` returns candles (default: the market data engine); only the last
    :data:`BARS` are used. ``progress(done, total)`` is called after each symbol, from worker threads;
    ``cancel()`` is polled between symbols, and a cancelled scan returns what it has. Raises ``ValueError``
    for bad input before anything is fetched and :class:`ScanBusy` if another scan is running.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"unsupported timeframe {timeframe!r}; use one of {' '.join(TIMEFRAMES)}")
    symbols = list(dict.fromkeys(symbols))
    if len(symbols) > MAX_SYMBOLS:
        raise ValueError(f"{len(symbols)} symbols: a scan covers at most {MAX_SYMBOLS}")
    cls = INDICATORS.get(indicator)
    ok, why = is_screenable(cls)
    if not ok:
        raise ValueError(f"{indicator} can't be scanned: {why}")
    values = resolve_params(cls.params, params or {})
    filters = check_filters(cls, filters or [])
    fetch = fetch or default_fetch
    if not _lock.acquire(blocking=False):
        raise ScanBusy("a scan is already running")
    t0 = time.monotonic()
    rows: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    done = 0
    counter = threading.Lock()

    def one(sym):
        nonlocal done
        if cancel and cancel():
            return
        try:
            df = fetch(sym, timeframe).tail(BARS)
            if df.empty:
                raise ValueError("no data")
            out = cls.compute(OHLC.from_frame(df), **values)
            row = {}
            for k in cls.outputs:
                v = float(np.asarray(out[k])[-1])
                row[k] = (v == 1.0) if k in cls.alerts else v  # an alert still warming up is False
            rows[sym] = row
        except Exception as exc:
            log.info("scanner skipped %s: %s", sym, exc)
            skipped[sym] = str(exc) or type(exc).__name__
        with counter:
            done += 1
            if progress:
                progress(done, len(symbols))

    try:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, symbols))
        cancelled = bool(cancel and cancel()) and done < len(symbols)
    finally:
        _lock.release()
    kept = [{"symbol": s, **r} for s, r in ((s, rows[s]) for s in symbols if s in rows) if _passes(r, filters)]
    table = pd.DataFrame(kept, columns=["symbol", *cls.outputs])
    return ScanResult(table, len(rows), len(table), {s: skipped[s] for s in symbols if s in skipped},
                      time.monotonic() - t0, cancelled)
