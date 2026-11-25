# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Economic calendar (#167): CPI, rate decisions, jobs data and other market-moving releases.

Sources, none of which needs a key:

* the free weekly calendar feed of Forex Factory (``nfs.faireconomy.media``): this week's releases of every country
  with their impact (high / medium / low), forecast and previous value;
* the Fed's published FOMC decision days of 2026 and the rule for the US jobs report (the third Friday after the
  week that contains the 12th), which reach further ahead than the feed. Those are marked ``approx``.

The feed only serves the current week, so every fetch is merged into a file in the cache folder: the calendar
grows with use and markers and the backtest rule see the weeks the app has been running. Nothing here raises: a
source that is down is skipped.

``flat_mask`` / ``apply_flat`` implement the option to keep a strategy flat around events (backtests).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tradebot.config import load_config, os_cache_dir

log = logging.getLogger(__name__)

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
USER_AGENT = "tradebot (open-source trading app)"
TIMEOUT = 15.0
CACHE_SECONDS = 3600.0
KINDS = ("rates", "cpi", "jobs", "other")  #: the kinds that get their own chart letter, most important first
IMPACTS = ("low", "medium", "high")
NEW_YORK = ZoneInfo("America/New_York")

#: the decision days of the Federal Open Market Committee in 2026 (the second day of each meeting, 14:00 New York)
FOMC_2026 = (date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17), date(2026, 7, 29),
             date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9))


@dataclass(frozen=True, order=True)
class EconEvent:
    time: datetime  #: UTC
    title: str
    country: str = ""  #: currency code of the feed ("USD", "EUR", ...)
    impact: str = "high"  #: low, medium or high
    kind: str = "other"  #: one of :data:`KINDS`
    forecast: str = ""
    previous: str = ""
    approx: bool = False  #: from a rule or schedule, not from the feed: the date may shift

    @property
    def key(self) -> tuple:
        return (self.time, self.title, self.country)


def classify(title: str) -> str:
    """The kind of a release from its title: ``rates``, ``cpi``, ``jobs`` or ``other``."""
    t = title.lower()
    if any(w in t for w in ("interest rate", "fomc", "federal funds", "rate decision", "rate statement",
                            "monetary policy", "main refinancing", "cash rate", "bank rate")):
        return "rates"
    if "cpi" in t or "consumer price" in t or "inflation rate" in t:
        return "cpi"
    if any(w in t for w in ("non-farm", "nonfarm", "unemployment rate", "employment change", "payroll",
                            "adp ")):
        return "jobs"
    return "other"


def _impact(raw: str) -> str:
    raw = str(raw or "").strip().lower()
    return raw if raw in IMPACTS else ""


def parse_feed(raw) -> list[EconEvent]:
    """The events of a Forex Factory calendar feed (a list of dicts); rows without a date, or holidays, are skipped."""
    out = []
    for row in raw if isinstance(raw, list) else []:
        try:
            impact = _impact(row.get("impact"))
            when = datetime.fromisoformat(str(row["date"]).replace("Z", "+00:00"))
            title = str(row["title"]).strip()
        except (KeyError, ValueError, AttributeError):
            continue
        if not impact or not title:  # "Holiday" rows carry no impact
            continue
        when = (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
        out.append(EconEvent(when, title, str(row.get("country", "")), impact, classify(title),
                             str(row.get("forecast") or ""), str(row.get("previous") or "")))
    return out


def jobs_report_day(year: int, month: int) -> date:
    """US employment situation release of ``month``: the third Friday after the week (Sunday to Saturday) that
    contains the 12th."""
    twelfth = date(year, month, 12)
    saturday = twelfth + timedelta(days=(5 - twelfth.weekday()) % 7)  # the week's Saturday (Monday is 0)
    day = saturday + timedelta(days=20)
    return day + timedelta(days=7) if (day.month, day.day) == (1, 1) else day  # not on New Year's Day


def _at_new_york(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NEW_YORK).astimezone(timezone.utc)


def scheduled(start: date, end: date) -> list[EconEvent]:
    """FOMC decisions (2026) and US jobs reports between ``start`` and ``end`` from the rules above."""
    out = [EconEvent(_at_new_york(d, 14), "FOMC Federal Funds Rate decision", "USD", "high", "rates", approx=True)
           for d in FOMC_2026 if start <= d <= end]
    y, m = (start.year - 1, 12) if start.month == 1 else (start.year, start.month - 1)  # a report comes a month on
    while (y, m) <= (end.year, end.month):
        d = jobs_report_day(y, m)
        if start <= d <= end:
            out.append(EconEvent(_at_new_york(d, 8, 30), "Non-Farm Employment Change", "USD", "high", "jobs",
                                 approx=True))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# -- fetching and the cache file -----------------------------------------------------------------------
def fetch_feed() -> list[EconEvent]:
    """This week's calendar from the feed. The seam tests replace; raises on network errors."""
    req = urllib.request.Request(FEED_URL, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 - fixed https endpoint
        return parse_feed(json.loads(resp.read().decode("utf-8")))


def _path():
    return os_cache_dir() / "economic_calendar.json"


def _read() -> tuple[list[EconEvent], float]:
    try:
        data = json.loads(_path().read_text())
        events = [EconEvent(datetime.fromisoformat(e[0]), *e[1:7], bool(e[7])) for e in data["events"]]
        return events, float(data.get("fetched", 0))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return [], 0.0


def _write(events: list[EconEvent], fetched: float) -> None:
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"fetched": fetched, "events": [
            [e.time.isoformat(), e.title, e.country, e.impact, e.kind, e.forecast, e.previous, e.approx] for e in events]}))
    except OSError as exc:
        log.debug("could not cache the economic calendar: %s", exc)


def load(start=None, end=None, fetch: Callable[[], list[EconEvent]] | None = None, refresh: bool = False,
         min_impact: str = "low", max_age: float = CACHE_SECONDS) -> list[EconEvent]:
    """The events from ``start`` to ``end`` (dates or timestamps; default: a month back to three months ahead),
    oldest first. Feed events beat the scheduled ones they duplicate; the feed is asked at most once an hour."""
    now = datetime.now(timezone.utc)
    lo = pd.Timestamp(start if start is not None else now - timedelta(days=30))
    hi = pd.Timestamp(end if end is not None else now + timedelta(days=90))
    lo = lo.tz_localize("UTC") if lo.tzinfo is None else lo.tz_convert("UTC")
    hi = hi.tz_localize("UTC") if hi.tzinfo is None else hi.tz_convert("UTC")
    cached, fetched = _read()
    if refresh or time.time() - fetched >= max_age:
        try:
            fresh = (fetch or fetch_feed)()
            by_key = {e.key: e for e in cached}
            by_key.update({e.key: e for e in fresh})
            cached, fetched = sorted(by_key.values()), time.time()
            _write(cached, fetched)
        except Exception as exc:  # noqa: BLE001 - offline or the feed changed: the cache and the schedule remain
            log.info("economic calendar feed unavailable: %s", exc)
    merged = list(cached)
    for s in scheduled(lo.date(), hi.date()):
        # the feed's own entry of the same kind on that day replaces the schedule's
        if not any(e.kind == s.kind and e.country == s.country and abs((e.time - s.time).total_seconds()) < 86400
                   and not e.approx for e in cached):
            merged.append(s)
    rank = IMPACTS.index(min_impact)
    return sorted(e for e in merged if lo <= e.time <= hi and IMPACTS.index(e.impact) >= rank)


# -- the "stay flat around events" rule -----------------------------------------------------------------
def settings() -> tuple[float, str]:
    """``([events] flat_around_hours, flat_impact)`` of config.toml: 0 (off) and ``"high"`` by default."""
    cfg = load_config().get("events", {})
    try:
        hours = max(0.0, float(cfg.get("flat_around_hours", 0)))
    except (TypeError, ValueError):
        hours = 0.0
    impact = str(cfg.get("flat_impact", "high")).lower()
    return hours, impact if impact in IMPACTS else "high"


def _step(index: pd.DatetimeIndex) -> pd.Timedelta:
    return pd.Series(index[1:] - index[:-1]).median() if len(index) > 1 else pd.Timedelta(days=1)


def flat_mask(index: pd.DatetimeIndex, events: list[EconEvent], hours: float) -> pd.Series:
    """True for the candles that touch the window from ``hours`` before to ``hours`` after an event. A candle
    stamped ``t`` covers ``[t, t + step)``."""
    idx = pd.DatetimeIndex(index)
    utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    mask = np.zeros(len(utc), dtype=bool)
    if hours > 0 and events and len(utc):
        opens = utc.as_unit("ns").asi8
        closes = opens + _step(utc).value
        pad = int(hours * 3600 * 1e9)
        for e in events:
            t = pd.Timestamp(e.time).as_unit("ns").value
            mask |= (opens <= t + pad) & (closes > t - pad)
    return pd.Series(mask, index=index, name="event_flat")


def apply_flat(positions: pd.Series, flat: pd.Series) -> pd.Series:
    """Positions with every candle inside ``flat`` forced to 0: open positions are closed before the event and
    nothing is opened until it is over."""
    pos = positions.to_numpy(dtype=float).copy()
    pos[flat.reindex(positions.index, fill_value=False).to_numpy(dtype=bool)] = 0.0
    return pd.Series(pos, index=positions.index, name=positions.name)


def event_flat(df: pd.DataFrame, hours: float, impact: str = "high",
               fetch: Callable[[], list[EconEvent]] | None = None) -> pd.Series | None:
    """The flat mask for the candles of ``df``, or None when the rule is off or no events are known there."""
    if hours <= 0 or len(df) == 0:
        return None
    pad = timedelta(hours=hours)
    events = load(pd.Timestamp(df.index[0]) - pad, pd.Timestamp(df.index[-1]) + pad, fetch=fetch, min_impact=impact)
    return flat_mask(df.index, events, hours) if events else None


def format_event(e: EconEvent) -> str:
    extra = " · ".join(p for p in (f"forecast {e.forecast}" if e.forecast else "",
                                   f"previous {e.previous}" if e.previous else "") if p)
    return f"{e.country} {e.title}" + (f" ({extra})" if extra else "")
