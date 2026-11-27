# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Several bots at once (#157): bot definitions in ``bots.toml``, shared account limits, one scheduler.

The bot set lives in ``bots.toml`` next to ``config.toml`` (``$TRADEBOT_BOTS`` overrides the path), so it survives a
reboot: ``tradebot bots run`` starts every enabled bot again (run it from a systemd unit, a login item or cron's
``@reboot``). The file is hand-editable; ``tradebot bots add``/``remove`` and the app's Bots page rewrite it whole::

    [limits]                      # shared by every bot of one manager; 0 or missing: off
    max_total_exposure = 500      # quote value of all open positions together, after an entry
    max_open_positions = 3        # bots holding a position at the same time
    max_daily_loss = 50           # realised + open loss of all bots in one UTC day: every bot halts until tomorrow
    close_on_halt = false         # also sell the open positions when that daily limit halts them

    [[bots]]
    name = "eth-trend"
    symbol = "ETH/USDT"
    algorithm = "sma_cross"
    broker = "binance"
    timeframe = "1h"
    stake = 50
    mode = "dry"                  # dry (nothing is sent), paper (the simulated account) or live (real orders)
    params = { fast = 10, slow = 30 }
    risk = { max_drawdown = 10 }  # the per-bot limits of `tradebot bot run`, see tradebot.live.risk

``live`` bots only run when the manager allows it (``tradebot bots run --live``); the desktop app never does.

Every running bot writes its state (position, P&L, the last events, a heartbeat) to ``bots/<name>.json`` in the config
folder after each step and once a minute, so a dashboard in another process can show it, a restarted bot resumes its
position, and the same bot is never run by two processes. A ``<name>.stop`` file next to it asks the process running
the bot to stop it (``tradebot bots stop``).

One scheduler thread steps all bots, each just after its candle closes, so bots that share a broker (the paper account)
never trade at the same time.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import tomllib
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tradebot import notify
from tradebot.config import config_path, os_config_dir
from tradebot.core.timeframes import to_timedelta
from tradebot.live.bot import BotEvent, TradingBot
from tradebot.live.risk import RiskLimits, check_limit, check_stake

log = logging.getLogger(__name__)

MODES = ("dry", "paper", "live")
HEARTBEAT = 60.0  #: seconds between state writes of an idle running bot
STALE = 3 * HEARTBEAT  #: a state file whose heartbeat is older than this belongs to no running process
EVENTS_KEPT = 30


def bots_path() -> Path:
    return Path(os.environ["TRADEBOT_BOTS"]).expanduser() if os.environ.get("TRADEBOT_BOTS") \
        else config_path().with_name("bots.toml")


def state_dir() -> Path:
    return os_config_dir() / "bots"


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "bot"


# --- definitions -----------------------------------------------------------------------------------------
@dataclass
class BotSpec:
    """One bot of the set: what ``tradebot bot run`` takes on its command line."""

    name: str
    symbol: str
    algorithm: str
    broker: str = "binance"
    timeframe: str = "1h"
    stake: float = 100.0
    mode: str = "dry"  #: dry, paper or live
    params: dict[str, float] = field(default_factory=dict)
    members: list[str] | None = None
    entry: str | None = None
    exit: str | None = None
    fills: str | None = None
    sizing: str | None = None
    sandbox: bool = False  #: the exchange testnet (ccxt brokers)
    enabled: bool = True  #: started by ``tradebot bots run`` (and the app's Start all)
    risk: dict[str, float] = field(default_factory=dict)  #: :class:`RiskLimits` fields

    def __post_init__(self):
        self.name, self.symbol = str(self.name).strip(), str(self.symbol).strip()
        if not self.name:
            raise ValueError("a bot needs a name")
        if not self.symbol or not str(self.algorithm).strip():
            raise ValueError(f"bot {self.name!r} needs a symbol and an algorithm")
        if self.mode not in MODES:
            raise ValueError(f"bot {self.name!r}: mode must be one of {', '.join(MODES)}, not {self.mode!r}")
        try:
            self.stake = check_stake(self.stake)
        except ValueError as e:
            raise ValueError(f"bot {self.name!r}: {e}") from None
        try:
            to_timedelta(self.timeframe)
        except Exception:
            raise ValueError(f"bot {self.name!r}: unknown timeframe {self.timeframe!r}") from None
        # a share may be kept as percent text ("50%"); the algorithm reads it (build_algorithm checks every value)
        self.params = {str(k): v.strip() if isinstance(v, str) and v.strip().endswith("%") else float(v)
                       for k, v in (self.params or {}).items()}
        known = {f.name for f in fields(RiskLimits)}
        unknown = set(self.risk or {}) - known
        if unknown:
            raise ValueError(f"bot {self.name!r}: unknown risk limit(s) {', '.join(sorted(unknown))}")
        try:
            self.limits()
        except ValueError as e:
            raise ValueError(f"bot {self.name!r}: {e}") from None

    @property
    def broker_name(self) -> str:
        return "paper" if self.mode == "paper" else self.broker

    def limits(self) -> RiskLimits:
        """The per-bot limits; always an object, so the kill switch applies to every bot."""
        return RiskLimits(**(self.risk or {}))

    def build_algorithm(self):
        from tradebot.algorithms import build_algorithm
        from tradebot.algorithms.sizing import PREFIX, sized

        size = {k: v for k, v in self.params.items() if k.startswith(PREFIX)}
        algo = build_algorithm(self.algorithm, {k: v for k, v in self.params.items() if k not in size},
                               self.members, self.entry, self.exit, self.fills)
        return sized(algo, self.sizing, **size)

    def describe(self) -> str:
        args = " ".join(f"{k}={v:g}" for k, v in self.params.items())
        return f"{self.algorithm}{' ' + args if args else ''}"

    @classmethod
    def from_dict(cls, data: dict) -> BotSpec:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"bot {data.get('name', '?')!r}: unknown key(s) {', '.join(sorted(unknown))}")
        missing = [k for k in ("name", "symbol", "algorithm") if k not in data]
        if missing:
            raise ValueError(f"bot {data.get('name', '?')!r}: missing {', '.join(missing)}")
        return cls(**data)

    def to_dict(self) -> dict:
        out = {}
        defaults = BotSpec("x", "x", "x")
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("name", "symbol", "algorithm") or value != getattr(defaults, f.name):
                out[f.name] = value
        return out


@dataclass
class AccountLimits:
    """Limits all bots of one manager share; 0/None switches one off."""

    max_total_exposure: float | None = None  #: quote value of every open position together, after an entry
    max_open_positions: int = 0  #: bots holding a position at once
    max_daily_loss: float | None = None  #: realised + open loss of all bots in one UTC day
    close_on_halt: bool = False  #: sell the open positions when the daily loss halts the bots

    def __post_init__(self):
        self.max_total_exposure = check_limit("max_total_exposure", self.max_total_exposure) or None
        self.max_daily_loss = check_limit("max_daily_loss", self.max_daily_loss) or None
        positions = check_limit("max_open_positions", self.max_open_positions) or 0
        if positions != int(positions):
            raise ValueError(f"risk limit max_open_positions: {self.max_open_positions!r} must be a whole number")
        self.max_open_positions = int(positions)
        self.close_on_halt = bool(self.close_on_halt)

    def active(self) -> bool:
        return bool(self.max_total_exposure or self.max_open_positions or self.max_daily_loss)

    def describe(self) -> str:
        parts = []
        if self.max_total_exposure:
            parts.append(f"exposure ≤ {self.max_total_exposure:,.0f}")
        if self.max_open_positions:
            parts.append(f"≤ {self.max_open_positions} positions")
        if self.max_daily_loss:
            parts.append(f"daily loss ≤ {self.max_daily_loss:,.0f}")
        return " · ".join(parts) or "none"


@dataclass
class BotSet:
    bots: list[BotSpec] = field(default_factory=list)
    limits: AccountLimits = field(default_factory=AccountLimits)

    def get(self, name: str) -> BotSpec:
        for b in self.bots:
            if b.name == name:
                return b
        raise KeyError(f"no bot named {name!r}")

    def put(self, spec: BotSpec, old_name: str | None = None) -> None:
        """Add ``spec`` or replace the bot named ``old_name`` (default: its own name), keeping its place."""
        key = old_name or spec.name
        if spec.name != key and any(b.name == spec.name for b in self.bots):
            raise ValueError(f"there already is a bot named {spec.name!r}")
        for i, b in enumerate(self.bots):
            if b.name == key:
                self.bots[i] = spec
                return
        self.bots.append(spec)

    def remove(self, name: str) -> None:
        self.get(name)
        self.bots = [b for b in self.bots if b.name != name]


def load_bots(path: Path | None = None) -> BotSet:
    """The bot set from ``bots.toml``; a missing file is an empty set. Raises ValueError on a broken file."""
    path = path or bots_path()
    if not path.is_file():
        return BotSet()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"can't read {path}: {exc}") from None
    limits = data.get("limits", {})
    unknown = set(limits) - {f.name for f in fields(AccountLimits)}
    if unknown:
        raise ValueError(f"{path}: unknown limit(s) {', '.join(sorted(unknown))}")
    try:
        bots = [BotSpec.from_dict(b) for b in data.get("bots", [])]
        account = AccountLimits(**limits)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from None
    names = [b.name for b in bots]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"{path}: more than one bot named {', '.join(sorted(dup))}")
    return BotSet(bots, account)


def _toml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:g}" if isinstance(value, float) and value == int(value) and abs(value) < 1e15 else repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a JSON string is a valid TOML basic string
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, dict):
        key = lambda k: k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)  # noqa: E731
        return "{ " + ", ".join(f"{key(k)} = {_toml(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"can't write {value!r} to TOML")


def save_bots(bots: BotSet, path: Path | None = None) -> Path:
    """Write the whole set (atomically); comments in a hand-edited file are not kept."""
    path = path or bots_path()
    lines = ["# tradebot bots (#157): `tradebot bots run` starts the enabled ones; see docs/bots.md", ""]
    limits = {k: v for k, v in asdict(bots.limits).items() if v}
    lines += ["[limits]", *(f"{k} = {_toml(v)}" for k, v in limits.items()), ""]
    for b in bots.bots:
        lines.append("[[bots]]")
        lines += [f"{k} = {_toml(v)}" for k, v in b.to_dict().items() if v is not None]
        lines.append("")
    text = "\n".join(lines)
    tomllib.loads(text)  # never write a file that can't be read back
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


# --- shared limits ---------------------------------------------------------------------------------------
class AccountGuard:
    """The :class:`AccountLimits` over the bots ``bots()`` returns (TradingBot objects)."""

    def __init__(self, limits: AccountLimits, bots: Callable[[], list[TradingBot]]):
        self.limits, self._bots = limits, bots
        self._day = None
        self._day_start = 0.0
        self.halted: str | None = None

    @property
    def close_on_halt(self) -> bool:
        return self.limits.close_on_halt

    def pnl(self) -> float:
        return sum(pnl(b)[0] + pnl(b)[1] for b in self._bots())

    def check(self, bot: TradingBot | None = None, now: datetime | None = None) -> str | None:
        """The shared halt message, or None while the bots may trade."""
        now = now or datetime.now(timezone.utc)
        total = self.pnl()
        if not math.isfinite(total):  # bad price data: no trading, and the day's starting point stays valid
            return "the bots' profit is not a number (bad price data); no trading until prices are valid again"
        if self._day != now.date():
            self._day, self._day_start, self.halted = now.date(), total, None  # a new day, a new allowance
        lim = self.limits.max_daily_loss
        if self.halted is None and lim and self._day_start - total >= lim:
            self.halted = f"all bots lost {self._day_start - total:.2f} today (shared limit {lim:g})"
            log.warning("account guard: %s", self.halted)
            notify.notify("risk", "Risk guard: all bots", self.halted)
        return self.halted

    def check_entry(self, bot: TradingBot, value: float) -> str | None:
        others = [b for b in self._bots() if b is not bot and b.holding > 0]
        lim = self.limits
        if lim.max_open_positions and len(others) >= lim.max_open_positions:
            return f"{len(others)} bots already hold a position (shared limit {lim.max_open_positions})"
        exposure = sum(b.holding * (b.last_price or b.entry_price) for b in others)
        if lim.max_total_exposure and exposure + value > lim.max_total_exposure:
            return f"all bots' exposure {exposure + value:.2f} above the shared limit {lim.max_total_exposure:g}"
        return None


def pnl(bot: TradingBot) -> tuple[float, float]:
    """``(realised, open)`` profit of one bot in the quote currency."""
    open_ = bot.holding * (bot.last_price - bot.entry_price) if bot.holding > 0 and bot.last_price else 0.0
    return bot.realized, open_


# --- running -----------------------------------------------------------------------------------------------
@dataclass
class BotStatus:
    """One row of the dashboard."""

    name: str
    symbol: str
    strategy: str
    timeframe: str
    mode: str
    state: str  #: stopped, running, halted, error or elsewhere (run by another process)
    position: float = 0.0
    entry: float = 0.0
    last: float = 0.0
    open_pnl: float = 0.0
    realized: float = 0.0
    last_event: str = ""
    last_time: str = ""
    next_check: str = ""
    error: str = ""
    enabled: bool = True
    events: list[dict] = field(default_factory=list)

    @property
    def pnl(self) -> float:
        return self.realized + self.open_pnl

    @property
    def exposure(self) -> float:
        return self.position * (self.last or self.entry)


def event_dict(e: BotEvent) -> dict:
    return {"time": e.time.isoformat(timespec="seconds"), "symbol": e.symbol, "action": e.action,
            "target": round(e.target, 4), "price": e.price, "amount": e.amount, "note": e.note,
            "order": e.order.id if e.order else ""}


def event_text(e: dict) -> str:
    if e["action"] == "hold":
        return e["note"] or f"hold at {e['price']:.6g}"
    text = f"{e['action']} {e['amount']:.6g} at {e['price']:.6g}"
    return f"{text} · {e['note']}" if e.get("note") else text


class _Runner:
    def __init__(self, spec: BotSpec):
        self.spec = spec
        self.bot: TradingBot | None = None
        self.next_due = 0.0
        self.last_beat = 0.0
        self.error = ""
        self.events: deque[dict] = deque(maxlen=EVENTS_KEPT)
        self.stepping = False

    @property
    def running(self) -> bool:
        return self.bot is not None


class BotManager:
    """Runs a :class:`BotSet`: start/stop single bots, one scheduler loop, the dashboard's snapshot.

    ``broker_factory(name, sandbox)`` builds brokers (default :func:`tradebot.brokers.create_broker`); one instance
    per name is shared by the bots, so the paper account is loaded once. ``allow_live``: live-mode bots may start.
    """

    def __init__(self, bots: BotSet, broker_factory: Callable | None = None, allow_live: bool = False,
                 state_path: Path | None = None, journal_factory: Callable | None = None,
                 clock: Callable[[], float] = time.time, on_event: Callable[[str, BotEvent], None] | None = None):
        self.bots, self.allow_live = bots, allow_live
        self._broker_factory = broker_factory or _default_broker
        self._journal_factory = journal_factory
        self.state_dir = state_path or state_dir()
        self.clock, self.on_event = clock, on_event
        self._brokers: dict = {}
        self._lock = threading.RLock()
        self._runners: dict[str, _Runner] = {b.name: _Runner(b) for b in bots.bots}
        self.guard = AccountGuard(bots.limits, lambda: [r.bot for r in self._runners.values() if r.bot])
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- the set ------------------------------------------------------------------------------------------
    def reload(self, bots: BotSet) -> None:
        """A changed set (edited in the app): running bots keep running unless removed or redefined."""
        with self._lock:
            new = {}
            for spec in bots.bots:
                r = self._runners.get(spec.name)
                if r is not None and r.running and r.spec != spec:
                    self.stop(spec.name)
                r = r or _Runner(spec)
                r.spec = spec
                new[spec.name] = r
            for name in set(self._runners) - set(new):
                self.stop(name)
            self._runners = new
            self.bots = bots
            self.guard.limits = bots.limits

    def names(self) -> list[str]:
        return list(self._runners)

    def is_running(self, name: str) -> bool:
        return self._runners[name].running

    # -- start / stop ------------------------------------------------------------------------------------
    def start(self, name: str) -> None:
        """Start one bot; raises ValueError (with the reason) when it may not or can't."""
        with self._lock:
            r = self._runners.get(name)
            if r is None:
                raise ValueError(f"no bot named {name!r}")
            if r.running:
                return
            spec = r.spec
            if spec.mode == "live" and not self.allow_live:
                raise ValueError(f"{name} is a live bot: live bots run from the terminal with "
                                 f"`tradebot bots run --live`")
            other = self.elsewhere(name)
            if other:
                raise ValueError(f"{name} already runs in another process (pid {other.get('pid')})")
            if spec.mode != "dry":
                for o in self._runners.values():
                    if o.running and o.spec.mode == spec.mode and o.spec.broker_name == spec.broker_name \
                            and o.spec.symbol.split("/")[0] == spec.symbol.split("/")[0]:
                        raise ValueError(f"{name} and {o.spec.name} would trade {spec.symbol.split('/')[0]} on the "
                                         f"same {spec.broker_name} account; run one of them")
            key = (spec.broker_name, spec.sandbox)
            if key not in self._brokers:
                self._brokers[key] = self._broker_factory(spec.broker_name, spec.sandbox)
            broker = self._brokers[key]
            if spec.mode == "live" and not getattr(broker, "can_trade", True):
                raise ValueError(f"{spec.broker} is read-only")
            try:
                algo = spec.build_algorithm()
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{name}: {exc}") from None
            bot = TradingBot(broker, algo, spec.symbol, spec.timeframe, spec.stake, dry_run=spec.mode == "dry",
                             risk=spec.limits(), account=self.guard,
                             journal=self._journal_factory() if self._journal_factory else None)
            saved = self._read_state(name)
            if saved and saved.get("mode") == spec.mode and saved.get("symbol") == spec.symbol:  # resume
                bot.holding, bot.fraction = float(saved.get("position", 0)), float(saved.get("fraction", 0))
                bot.entry_price, bot.realized = float(saved.get("entry", 0)), float(saved.get("realized", 0))
                bot.last_price = float(saved.get("last", 0))
                r.events.extend(saved.get("events", [])[-EVENTS_KEPT:])
            r.bot, r.error, r.next_due = bot, "", self.clock()
            try:  # an old stop request; like the state file, a broken state folder must not keep the bot from running
                self.stop_request_path(name).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not clear the stop request of bot %s: %s", name, exc)
            self._write_state(r)
            log.info("bot %s started (%s %s %s, %s)", name, spec.symbol, spec.algorithm, spec.timeframe, spec.mode)

    def stop(self, name: str) -> None:
        with self._lock:
            r = self._runners.get(name)
            if r is None or not r.running:
                return
            self._write_state(r, "stopped")
            r.bot = None
            log.info("bot %s stopped", name)

    def start_all(self, enabled_only: bool = True) -> dict[str, str]:
        """Start every (enabled) bot; returns name -> why it didn't start."""
        failed = {}
        for name, r in list(self._runners.items()):
            if enabled_only and not r.spec.enabled:
                continue
            try:
                self.start(name)
            except Exception as exc:  # one broken bot doesn't keep the others from starting
                failed[name] = str(exc)
        return failed

    def stop_all(self) -> None:
        for name in list(self._runners):
            self.stop(name)

    def running(self) -> list[str]:
        return [n for n, r in self._runners.items() if r.running]

    # -- scheduling --------------------------------------------------------------------------------------
    def tick(self) -> list[tuple[str, BotEvent]]:
        """Step every bot whose candle has closed; handle stop requests and heartbeats. Returns the new events."""
        now = self.clock()
        out = []
        for name, r in list(self._runners.items()):
            if r.running and self.stop_request_path(name).exists():
                try:
                    self.stop_request_path(name).unlink(missing_ok=True)
                except OSError as exc:  # the stop must happen anyway
                    log.warning("could not remove the stop request of bot %s: %s", name, exc)
                self.stop(name)
                continue
            if not r.running:
                continue
            if now >= r.next_due:
                ev = self._step(r)
                if ev is not None:
                    out.append((name, ev))
            elif now - r.last_beat >= HEARTBEAT:
                self._write_state(r)
        return out

    def _step(self, r: _Runner) -> BotEvent | None:
        bot = r.bot
        step = to_timedelta(r.spec.timeframe).total_seconds()
        now = self.clock()
        r.next_due = now - now % step + step + 2  # just after the next candle closes
        try:
            ev = bot.step()
            r.error = ""
        except Exception as exc:
            log.exception("bot %s step failed; retrying next candle", r.spec.name)
            r.error = f"{type(exc).__name__}: {exc}"
            notify.notify("error", f"Bot step failed: {r.spec.name}", "See the log; retrying next candle.")
            ev = None
        else:
            del bot.events[:-EVENTS_KEPT]  # the runner keeps the history
            r.events.append(event_dict(ev))
            notify.bot_event(ev)
            if self.on_event:
                self.on_event(r.spec.name, ev)
        if r.bot is bot:  # not stopped meanwhile
            self._write_state(r)
        return ev

    def run(self, max_ticks: int | None = None, stop: threading.Event | None = None, poll: float = 1.0) -> None:
        """The scheduler loop, until ``stop`` is set (or ``max_ticks`` passes, or no bot runs any more)."""
        stop = stop or self._stop
        n = 0
        while not stop.is_set():
            self.tick()
            n += 1
            if (max_ticks is not None and n >= max_ticks) or not self.running():
                break
            stop.wait(poll)

    def start_loop(self) -> None:
        """Run the scheduler in a background thread (the app); it keeps going while no bot runs."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception:  # the loop must survive anything a bot does
                    log.exception("bot scheduler tick failed")
                self._stop.wait(1.0)

        self._thread = threading.Thread(target=loop, name="tradebot-bots", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the loop and every bot (their state is saved, so they resume next time)."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        self.stop_all()

    # -- state files -------------------------------------------------------------------------------------
    def state_path(self, name: str) -> Path:
        return self.state_dir / f"{_slug(name)}.json"

    def stop_request_path(self, name: str) -> Path:
        return self.state_dir / f"{_slug(name)}.stop"

    def _read_state(self, name: str) -> dict | None:
        try:
            data = json.loads(self.state_path(name).read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def elsewhere(self, name: str) -> dict | None:
        """The state of ``name`` when another live process runs it, else None."""
        s = self._read_state(name)
        if not s or s.get("state") == "stopped" or s.get("pid") == os.getpid():
            return None
        return s if self.clock() - float(s.get("heartbeat", 0)) < STALE else None

    def _write_state(self, r: _Runner, state: str | None = None) -> None:
        bot = r.bot
        if bot is None:
            return
        r.last_beat = self.clock()
        realized, open_ = pnl(bot)
        data = {"name": r.spec.name, "symbol": r.spec.symbol, "mode": r.spec.mode, "pid": os.getpid(),
                "heartbeat": r.last_beat, "state": state or self._state(r), "position": bot.holding,
                "fraction": bot.fraction, "entry": bot.entry_price, "last": bot.last_price, "realized": realized,
                "open_pnl": open_, "error": r.error, "next_check": r.next_due, "events": list(r.events)}
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            path = self.state_path(r.spec.name)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            log.warning("could not save the state of bot %s: %s", r.spec.name, exc)

    def _state(self, r: _Runner) -> str:
        if r.error:
            return "error"
        if self.guard.halted or r.events and r.events[-1].get("note", "").startswith("halted"):
            return "halted"
        return "running"

    # -- dashboard ---------------------------------------------------------------------------------------
    def snapshot(self) -> list[BotStatus]:
        out = []
        for name, r in list(self._runners.items()):
            spec = r.spec
            row = BotStatus(name, spec.symbol, spec.describe(), spec.timeframe, spec.mode, "stopped",
                            enabled=spec.enabled)
            bot = r.bot
            if bot is not None:
                realized, open_ = pnl(bot)
                row.state = self._state(r)
                row.position, row.entry, row.last = bot.holding, bot.entry_price, bot.last_price
                row.open_pnl, row.realized, row.error = open_, realized, r.error
                row.events = list(r.events)
                row.next_check = _hhmm(r.next_due)
            else:
                s = self.elsewhere(name) or self._read_state(name)
                if s and s.get("symbol") == spec.symbol and s.get("mode") == spec.mode:
                    if self.elsewhere(name):
                        row.state = "elsewhere"
                        row.next_check = _hhmm(float(s.get("next_check") or 0))
                    row.position, row.entry, row.last = (float(s.get(k) or 0) for k in ("position", "entry", "last"))
                    row.open_pnl, row.realized = float(s.get("open_pnl") or 0), float(s.get("realized") or 0)
                    row.error, row.events = s.get("error", "") if row.state != "stopped" else "", s.get("events", [])
            if row.events:
                row.last_event, row.last_time = event_text(row.events[-1]), row.events[-1]["time"][5:16].replace("T", " ")
            out.append(row)
        return out

    def summary(self, rows: list[BotStatus] | None = None) -> dict:
        rows = self.snapshot() if rows is None else rows
        live = [r for r in rows if r.state != "stopped"]
        return {"running": len(live), "bots": len(rows), "positions": sum(1 for r in live if r.position > 0),
                "exposure": sum(r.exposure for r in live), "pnl": sum(r.pnl for r in live),
                "halted": self.guard.halted, "limits": self.bots.limits.describe()}


def _hhmm(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%H:%M") if epoch else ""


def _default_broker(name: str, sandbox: bool = False):
    from tradebot.brokers import create_broker

    return create_broker(name, **({"sandbox": True} if sandbox else {}))


def request_stop(name: str, path: Path | None = None) -> Path:
    """Ask the process that runs ``name`` to stop it at its next tick (within a second or so)."""
    p = (path or state_dir()) / f"{_slug(name)}.stop"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("stop\n", encoding="utf-8")
    return p
