# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Background alerts (#156): price and indicator alerts per symbol, and scheduled scanner runs.

Three kinds, stored together in ``alerts.json`` in the OS config folder (shared by the app, the CLI and the
headless ``tradebot watch``; ``$TRADEBOT_ALERTS`` overrides the path):

* ``price``: ``BTC/USDT >= 100000``. Fires when the last close of the alert's timeframe reaches the level.
* ``indicator``: ``AAPL`` with ``rsi < 30`` (any scanner filter on an indicator's outputs or alerts).
* ``scan``: a scanner run over several symbols; only matches that are new since the last run are reported.

Every alert has ``every``: minutes between checks, or ``"close"`` for once per finished candle of its timeframe.
A price or indicator alert fires on the edge (the condition turns true) and then waits until it was false
again (``repeat``) or disables itself (default). Delivery uses :mod:`tradebot.notify`, kind ``alert`` for
price/indicator alerts and ``scanner`` for scan runs.

:func:`run_due` does one pass and is what both the app's timer and :func:`watch` call; data fetching and
sending are injectable, so it is testable without a network. Run only one watcher (the app's or
``tradebot watch``) at a time, or alerts fire twice.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from tradebot import scanner
from tradebot.config import os_config_dir

log = logging.getLogger(__name__)

KINDS = ("price", "indicator", "scan")
PRICE_OPS = (">=", "<=", ">", "<")
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
              "1d": 86400, "1w": 7 * 86400, "1mo": 30 * 86400}  #: candle length; "close" aligns to the epoch
_io = threading.Lock()


def alerts_path() -> Path:
    env = os.environ.get("TRADEBOT_ALERTS")
    return Path(env).expanduser() if env else os_config_dir() / "alerts.json"


@dataclass
class Alert:
    kind: str
    symbols: list[str]
    timeframe: str = "1m"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    op: str = ">="  #: price alerts
    value: float = 0.0  #: price alerts
    indicator: str = ""  #: indicator and scan alerts
    params: dict = field(default_factory=dict)
    filters: list = field(default_factory=list)
    every: int | str = 1  #: minutes, or "close"
    repeat: bool = False
    enabled: bool = True
    armed: bool = True  #: False while the condition still holds after it fired (repeat alerts)
    last_run: float = 0.0  #: epoch seconds
    last_matches: list = field(default_factory=list)  #: scan: the symbols reported last time
    last_fired: float = 0.0
    note: str = ""

    def describe(self) -> str:
        if self.kind == "price":
            return f"{self.symbols[0]} {self.timeframe} close {self.op} {self.value:g}"
        conds = " & ".join(_cond(f) for f in self.filters) or "any"
        target = self.symbols[0] if self.kind == "indicator" else f"{len(self.symbols)} symbols"
        return f"{target} {self.timeframe} {self.indicator}: {conds}"


def _cond(f: dict) -> str:
    if "alert" in f:
        return f["alert"] + ("" if f.get("state", True) else " off")
    return f"{f['output']} {f['op']} {f['value']}"


def validate(a: Alert) -> Alert:
    """Raise ``ValueError`` for an alert that could never run; returns it normalised."""
    if a.kind not in KINDS:
        raise ValueError(f"unknown alert kind {a.kind!r}; use one of {' '.join(KINDS)}")
    if a.timeframe not in scanner.TIMEFRAMES:
        raise ValueError(f"unsupported timeframe {a.timeframe!r}; use one of {' '.join(scanner.TIMEFRAMES)}")
    if not a.symbols or (a.kind != "scan" and len(a.symbols) != 1):
        raise ValueError("an alert needs one symbol" if a.kind != "scan" else "a scan alert needs symbols")
    if a.every != "close":
        try:
            a.every = int(a.every)
        except (TypeError, ValueError):
            raise ValueError(f"every: minutes or 'close', not {a.every!r}") from None
        if a.every < 1:
            raise ValueError("every must be at least 1 minute")
    if a.kind == "price":
        if a.op not in PRICE_OPS:
            raise ValueError(f"unknown operator {a.op!r}; use one of {' '.join(PRICE_OPS)}")
        try:
            a.value = float(a.value)
        except (TypeError, ValueError):
            raise ValueError(f"price: a number, not {a.value!r}") from None
        if not math.isfinite(a.value):
            raise ValueError(f"price: a finite number, not {a.value!r}")  # NaN never compares true: never fires
    else:
        from tradebot.indicators import INDICATORS

        try:
            cls = INDICATORS.get(a.indicator)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        a.filters = scanner.check_filters(cls, a.filters)
        if not a.filters:
            raise ValueError("needs at least one condition (a value filter or an indicator alert)")
    return a


# -- storage ---------------------------------------------------------------------------------------
def _read(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        log.warning("ignoring alerts file %s: %s", path, exc)
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def _write(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _set_aside_if_unreadable(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2))
    tmp.replace(path)  # atomic


def _set_aside_if_unreadable(path: Path) -> None:
    """An alerts file that isn't valid JSON (a typo from a hand edit) would be replaced by the next write and the
    user's alerts lost; move it to ``<name>.corrupt`` first, to be repaired by hand."""
    try:
        json.loads(path.read_text())
        return
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        pass
    backup = path.with_name(path.name + ".corrupt")
    if backup.exists():  # never overwrite an earlier backup
        backup = path.with_name(f"{path.name}.corrupt.{time.time_ns()}")
    path.replace(backup)
    log.warning("alerts file %s was unreadable; moved to %s, starting a new one", path, backup)


def load(path: Path | None = None) -> list[Alert]:
    """All stored alerts; a malformed entry is skipped."""
    known = {f.name for f in fields(Alert)}
    out = []
    for d in _read(path or alerts_path()):
        try:
            out.append(Alert(**{k: v for k, v in d.items() if k in known}))
        except TypeError:
            log.warning("skipping malformed alert %r", d)
    return out


def add(alert: Alert, path: Path | None = None) -> Alert:
    validate(alert)
    path = path or alerts_path()
    with _io:
        _write(path, _read(path) + [asdict(alert)])
    return alert


def remove(alert_id: str, path: Path | None = None) -> bool:
    path = path or alerts_path()
    with _io:
        items = _read(path)
        kept = [d for d in items if d.get("id") != alert_id]
        if len(kept) != len(items):
            _write(path, kept)
        return len(kept) != len(items)


def update(alert_id: str, path: Path | None = None, **changes) -> None:
    """Patch one stored alert (re-reading the file, so edits made elsewhere meanwhile are kept)."""
    path = path or alerts_path()
    with _io:
        items = _read(path)
        for d in items:
            if d.get("id") == alert_id:
                d.update(changes)
                _write(path, items)
                return


# -- evaluation ------------------------------------------------------------------------------------
def is_due(a: Alert, now: float) -> bool:
    if not a.enabled:
        return False
    if a.every == "close":
        step = TF_SECONDS[a.timeframe]
        return a.last_run == 0 or now // step > a.last_run // step
    return now - a.last_run >= int(a.every) * 60


def _price_hit(a: Alert, price: float) -> bool:
    return {">=": price >= a.value, "<=": price <= a.value, ">": price > a.value, "<": price < a.value}[a.op]


def check(a: Alert, fetch: Callable | None, notify_fn: Callable, now: float) -> dict:
    """Evaluate one alert; returns the state fields to store. ``notify_fn(kind, title, text)`` delivers."""
    fetch = fetch or scanner.default_fetch
    state: dict = {"last_run": now}
    if a.kind == "scan":
        res = scanner.scan(a.symbols, a.indicator, a.params, a.timeframe, a.filters, fetch=fetch)
        matching = [str(s) for s in res.table["symbol"]]
        new = [s for s in matching if s not in a.last_matches]
        if new:
            notify_fn("scanner", f"Scan {a.indicator} {a.timeframe}: {len(new)} new", ", ".join(new))
            state["last_fired"] = now
        if res.scanned:  # a run that got no data at all keeps the memory
            state["last_matches"] = matching
        return state
    sym = a.symbols[0]
    if a.kind == "price":
        df = fetch(sym, a.timeframe)
        if df.empty:
            raise ValueError(f"no data for {sym}")
        price = float(df["close"].iloc[-1])
        hit, text, title = _price_hit(a, price), f"{sym} {price:.6g} ({a.op} {a.value:g})", f"Price alert {sym}"
    else:
        res = scanner.scan([sym], a.indicator, a.params, a.timeframe, a.filters, fetch=fetch)
        if res.skipped:
            raise ValueError(next(iter(res.skipped.values())))
        hit, text, title = res.matched > 0, a.describe(), f"Indicator alert {sym}"
    if hit and a.armed:
        notify_fn("alert", title, text)
        state["last_fired"] = now
        state["armed" if a.repeat else "enabled"] = False
    elif not hit and not a.armed:
        state["armed"] = True
    return state


def run_due(fetch: Callable | None = None, notify_fn: Callable | None = None, now: float | None = None,
            path: Path | None = None) -> list[str]:
    """One pass over the stored alerts: run what is due, notify, store the new state. Returns log lines."""
    if notify_fn is None:
        from tradebot.notify import notify as notify_fn
    now = time.time() if now is None else now
    lines = []
    for a in load(path):
        if not is_due(a, now):
            continue
        try:
            state = check(a, fetch, notify_fn, now)
        except scanner.ScanBusy:
            continue  # try again next pass
        except Exception as exc:  # one dead symbol never stops the others
            log.warning("alert %s failed: %s", a.id, exc)
            lines.append(f"{a.id} {a.describe()}: failed: {exc}")
            state = {"last_run": now}
        else:
            if "last_fired" in state:
                lines.append(f"{a.id} {a.describe()}: fired")
        update(a.id, path, **state)
    return lines


def watch(poll: float = 30.0, stop: Callable[[], bool] | None = None, log_fn: Callable[[str], None] = print,
          **kw) -> None:
    """Headless loop (``tradebot watch``): :func:`run_due` every ``poll`` seconds until ``stop()`` or Ctrl-C."""
    stop = stop or (lambda: False)
    while not stop():
        for line in run_due(**kw):
            log_fn(f"{datetime.now(timezone.utc):%H:%M:%S} {line}")
        slept = 0.0
        while slept < poll and not stop():
            time.sleep(min(1.0, poll - slept))
            slept += 1.0


def parse_price(text: str) -> tuple[str, float]:
    """``">=100000"`` -> ``(">=", 100000.0)``; raises ``ValueError``."""
    t = text.strip()
    for op in (">=", "<=", ">", "<"):
        if t.startswith(op):
            try:
                value = float(t[len(op):])
            except ValueError:
                break
            if math.isfinite(value):
                return op, value
            break
    raise ValueError(f"{text!r}: use '>=100000', '<=90000', '>' or '<' and a number")
