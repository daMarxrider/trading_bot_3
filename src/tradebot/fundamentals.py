# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Fundamentals and corporate events of a stock (#165): valuation ratios, margins, growth, dividend yield, and
the dates of earnings, splits and dividends.

The free source is Yahoo Finance via yfinance. One fetch per symbol is cached as JSON in the OS cache folder
(:func:`~tradebot.config.os_cache_dir`, ``fundamentals/``) for a day, so a screener over a watchlist asks Yahoo
once. Nothing here raises: a symbol without such data (crypto, an index, a delisted ticker) or a failed request
gives an empty :class:`Snapshot` (``error`` says why); a stale cache entry is served when the refresh fails.

``[events]`` in config.toml sets the optional trading rule "no new position N days before earnings"::

    [events]
    earnings_blackout_days = 3   # default 0 = off
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from tradebot.config import load_config, os_cache_dir

log = logging.getLogger(__name__)

CACHE_SECONDS = 24 * 3600
KINDS = ("earnings", "split", "dividend")

#: fundamentals fields shown and screened: name -> (label, kind of number: "ratio" | "pct" | "money" | "text")
FIELDS = {
    "pe": ("P/E", "ratio"),
    "forward_pe": ("Forward P/E", "ratio"),
    "ps": ("P/S", "ratio"),
    "profit_margin": ("Profit margin", "pct"),
    "operating_margin": ("Operating margin", "pct"),
    "revenue_growth": ("Revenue growth", "pct"),
    "earnings_growth": ("Earnings growth", "pct"),
    "dividend_yield": ("Dividend yield", "pct"),
    "market_cap": ("Market cap", "money"),
}
#: the screener columns this module adds (plus ``days_to_earnings``)
SCREEN_COLUMNS = [f for f in FIELDS if f != "market_cap"] + ["days_to_earnings"]


@dataclass
class Fundamentals:
    name: str = ""
    sector: str = ""
    currency: str = ""
    pe: float | None = None
    forward_pe: float | None = None
    ps: float | None = None
    profit_margin: float | None = None
    operating_margin: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    dividend_yield: float | None = None  #: fraction of the price per year (0.02 = 2 %)
    market_cap: float | None = None

    def has_data(self) -> bool:
        return any(getattr(self, f) is not None for f in FIELDS)


@dataclass(frozen=True, order=True)
class Event:
    day: date
    kind: str  #: one of :data:`KINDS`
    label: str = ""  #: "Earnings", "Split 4:1", "Ex-dividend 0.24"
    value: float | None = None  #: split ratio / dividend per share

    @property
    def upcoming(self) -> bool:
        return self.day >= date.today()


@dataclass
class Snapshot:
    symbol: str
    fundamentals: Fundamentals = field(default_factory=Fundamentals)
    events: list[Event] = field(default_factory=list)
    fetched: float = 0.0  #: unix time of the fetch
    error: str = ""  #: why there is no data, "" when there is

    def earnings(self) -> list[date]:
        return [e.day for e in self.events if e.kind == "earnings"]

    def next_earnings(self, today: date | None = None) -> date | None:
        today = today or date.today()
        return min((d for d in self.earnings() if d >= today), default=None)

    def next_event(self, kind: str, today: date | None = None) -> Event | None:
        today = today or date.today()
        return min((e for e in self.events if e.kind == kind and e.day >= today), default=None)


def supported(symbol: str) -> bool:
    """Fundamentals exist for stocks only: not crypto pairs (``BTC/USDT``), pair symbols (``KO:PEP``) or indices."""
    return bool(symbol) and "/" not in symbol and ":" not in symbol and not symbol.startswith("^")


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _day(v) -> date | None:
    try:
        ts = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else ts.date()


def parse_info(info: dict) -> Fundamentals:
    """yfinance's ``Ticker.info`` -> :class:`Fundamentals`; missing or non-numeric entries stay None."""
    price = _num(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    rate = _num(info.get("dividendRate") or info.get("trailingAnnualDividendRate"))
    dy = rate / price if rate and price else _num(info.get("trailingAnnualDividendYield"))
    pe = _num(info.get("trailingPE"))
    return Fundamentals(
        name=str(info.get("longName") or info.get("shortName") or ""),
        sector=str(info.get("sector") or ""),
        currency=str(info.get("currency") or ""),
        pe=pe if pe and pe > 0 else None,  # a loss-making company has no meaningful P/E
        forward_pe=_num(info.get("forwardPE")),
        ps=_num(info.get("priceToSalesTrailing12Months")),
        profit_margin=_num(info.get("profitMargins")),
        operating_margin=_num(info.get("operatingMargins")),
        revenue_growth=_num(info.get("revenueGrowth")),
        earnings_growth=_num(info.get("earningsGrowth")),
        dividend_yield=dy,
        market_cap=_num(info.get("marketCap")),
    )


def parse_events(raw: dict) -> list[Event]:
    """The raw fetch -> events: ``earnings`` (dates), ``splits`` and ``dividends`` (date -> number),
    ``ex_dividend`` (the next ex-dividend date, if known)."""
    out: set[Event] = set()
    for d in raw.get("earnings") or []:
        if (day := _day(d)) is not None:
            out.add(Event(day, "earnings", "Earnings"))
    for d, v in (raw.get("splits") or {}).items():
        v = _num(v)
        if (day := _day(d)) is not None and v:
            out.add(Event(day, "split", f"Split {v:g}:1" if v >= 1 else f"Reverse split 1:{1 / v:g}", v))
    for d, v in (raw.get("dividends") or {}).items():
        v = _num(v)
        if (day := _day(d)) is not None and v:
            out.add(Event(day, "dividend", f"Ex-dividend {v:.4g}", v))
    if (day := _day(raw.get("ex_dividend"))) is not None and not any(e.kind == "dividend" and e.day == day for e in out):
        out.add(Event(day, "dividend", "Ex-dividend"))
    return sorted(out)


def fetch_raw(symbol: str) -> dict:
    """One round of yfinance requests; raises on a failed ``info`` request. The seam tests replace."""
    import yfinance as yf

    from tradebot.data.providers import YFinanceProvider

    t = yf.Ticker(YFinanceProvider.map_symbol(symbol))
    raw: dict = {"info": dict(t.info or {})}
    try:
        raw["earnings"] = [d for d in t.get_earnings_dates(limit=24).index]
    except Exception as exc:  # not every symbol has earnings dates; the rest is still useful
        log.debug("no earnings dates for %s: %s", symbol, exc)
    try:
        raw["splits"] = {d: v for d, v in t.splits.items()}
        raw["dividends"] = {d: v for d, v in t.dividends.items()}
    except Exception as exc:
        log.debug("no splits/dividends for %s: %s", symbol, exc)
    try:
        ex = (t.calendar or {}).get("Ex-Dividend Date")
        raw["ex_dividend"] = ex
        raw["earnings"] = list(raw.get("earnings", [])) + list((t.calendar or {}).get("Earnings Date") or [])
    except Exception as exc:
        log.debug("no calendar for %s: %s", symbol, exc)
    return raw


def _path(symbol: str):
    return os_cache_dir() / "fundamentals" / (re.sub(r"[^A-Za-z0-9._=-]", "_", symbol) + ".json")


def _encode(s: Snapshot) -> dict:
    return {"symbol": s.symbol, "fetched": s.fetched, "fundamentals": asdict(s.fundamentals),
            "events": [[e.day.isoformat(), e.kind, e.label, e.value] for e in s.events]}


def _decode(d: dict) -> Snapshot:
    f = Fundamentals(**{k: v for k, v in d.get("fundamentals", {}).items() if k in Fundamentals.__dataclass_fields__})
    ev = [Event(date.fromisoformat(a), b, c, v) for a, b, c, v in d.get("events", [])]
    return Snapshot(d["symbol"], f, ev, float(d.get("fetched", 0)))


def _read(symbol: str) -> Snapshot | None:
    try:
        return _decode(json.loads(_path(symbol).read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write(s: Snapshot) -> None:
    try:
        p = _path(s.symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_encode(s)), encoding="utf-8")
    except OSError as exc:
        log.debug("could not cache fundamentals of %s: %s", s.symbol, exc)


def load(symbol: str, fetch: Callable[[str], dict] | None = None, refresh: bool = False,
         max_age: float = CACHE_SECONDS) -> Snapshot:
    """Fundamentals and events of ``symbol``: from the cache while fresh, else fetched (and cached). Never raises."""
    if not supported(symbol):
        return Snapshot(symbol, error="no fundamentals for crypto, pairs or indices")
    cached = _read(symbol)
    if cached and not refresh and time.time() - cached.fetched < max_age:
        return cached
    try:
        raw = (fetch or fetch_raw)(symbol)
        snap = Snapshot(symbol, parse_info(raw.get("info") or {}), parse_events(raw), time.time())
    except Exception as exc:
        log.warning("fundamentals of %s unavailable: %s", symbol, exc)
        return cached or Snapshot(symbol, error=str(exc) or type(exc).__name__)
    if not snap.fundamentals.has_data() and not snap.events:
        return cached or Snapshot(symbol, error="no data from Yahoo Finance")
    _write(snap)
    return snap


def load_many(symbols: list[str], fetch: Callable[[str], dict] | None = None, workers: int = 8) -> dict[str, Snapshot]:
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(symbols, pool.map(lambda s: load(s, fetch), symbols)))


def screen_values(snap: Snapshot, today: date | None = None) -> dict[str, float]:
    """The screener columns of a snapshot (NaN where unknown); ``days_to_earnings``: to the next earnings date."""
    row = {k: (np.nan if getattr(snap.fundamentals, k) is None else getattr(snap.fundamentals, k)) for k in SCREEN_COLUMNS[:-1]}
    nxt = snap.next_earnings(today)
    row["days_to_earnings"] = np.nan if nxt is None else float((nxt - (today or date.today())).days)
    return row


# -- the "no new position before earnings" rule ---------------------------------------------------------

def blackout_days() -> int:
    """``[events] earnings_blackout_days`` in config.toml; 0 (off) by default."""
    try:
        return max(0, int(load_config().get("events", {}).get("earnings_blackout_days", 0)))
    except (TypeError, ValueError):
        return 0


def blackout_mask(index: pd.DatetimeIndex, earnings: list[date], days: int) -> pd.Series:
    """True for the candles within ``days`` calendar days before an earnings date (the day itself included)."""
    mask = np.zeros(len(index), dtype=bool)
    if days > 0 and earnings:
        idx = pd.DatetimeIndex(index)
        days_of = (idx.tz_localize(None) if idx.tz is not None else idx).normalize().to_numpy("datetime64[D]")
        for e in earnings:
            e64 = np.datetime64(e, "D")
            mask |= (days_of <= e64) & (days_of >= e64 - np.timedelta64(days, "D"))
    return pd.Series(mask, index=index, name="earnings_blackout")


def apply_blackout(positions: pd.Series, blackout: pd.Series) -> pd.Series:
    """Positions with no new entry inside the blackout: a candle that would go from flat (or from the other side)
    to a position stays flat. A position opened before the blackout is kept; the rule never forces an exit."""
    pos = positions.to_numpy(dtype=float).copy()
    block = blackout.reindex(positions.index, fill_value=False).to_numpy(dtype=bool)
    prev = 0.0
    for i in range(len(pos)):
        if block[i] and pos[i] != 0 and (prev == 0 or np.sign(pos[i]) != np.sign(prev)):
            pos[i] = 0.0
        prev = pos[i]
    return pd.Series(pos, index=positions.index, name=positions.name)


def earnings_blackout(df: pd.DataFrame, symbol: str, days: int, fetch: Callable[[str], dict] | None = None) -> pd.Series | None:
    """The blackout mask for the candles of ``df``, or None when the rule is off or no earnings dates are known."""
    if days <= 0 or not supported(symbol):
        return None
    dates = load(symbol, fetch).earnings()
    return blackout_mask(df.index, dates, days) if dates else None


def format_value(name: str, v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    kind = FIELDS[name][1]
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "money":
        return f"{v / 1e9:,.1f}B" if abs(v) >= 1e8 else f"{v / 1e6:,.1f}M"
    return f"{v:.1f}"


__all__ = ["Event", "Fundamentals", "Snapshot", "load", "load_many", "blackout_mask", "apply_blackout"]
