# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Live two-legged pairs trading (#117): long one asset and short the other at the same time.

Same rule as :func:`tradebot.backtest.pairs.pair_backtest` (:func:`~tradebot.backtest.pairs.next_side`), on
each *closed* candle. The stake is split by the hedge ratio at entry, ``stake / (1 + β)`` in leg A and
``stake · β / (1 + β)`` in leg B, and the amounts that really filled are kept until the exit. Dry-run is the
default.

The short leg is an ordinary SELL, so the broker needs margin or futures for it: pass the legs as the symbols
that broker can short (e.g. perpetual contracts ``BTC/USDT:USDT``). A spot account rejects the sale; the bot
then undoes the leg already placed and stops trading that step.

Remembering the open pair (#122). With a ``state_path`` (the CLI uses :func:`pair_state_path` for ``--live``)
the bot writes the pair (side, amount in each leg) to a JSON file, atomically, and picks it up after a restart.
Every trade is written ahead: the file records the trade as *pending* before the first order and after each
fill, and only a finished trade clears it. A restart that finds a pending trade (a crash mid-trade, or an undo
that failed) refuses to trade until someone has looked at the account and run ``tradebot bot pair-state``
with ``--keep`` or ``--forget``. On start the saved pair is also checked against the broker's positions
(:meth:`~tradebot.brokers.base.BaseBroker.leg_position`) where the broker can report them.

Fills: an order still open is polled (``get_order``) for ``fill_timeout`` seconds, then the rest is cancelled
and what filled counts. A partly filled first leg shrinks the second to keep the hedge; a partly filled close
leaves the rest open and the bot closes it at the next candle whatever the signal says. Anything whose outcome
the bot can't know (a timeout while placing an order, a failed undo, an order that can't be cancelled) halts
the bot with the pending trade kept in the file.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from tradebot import notify
from tradebot.analysis.pairs import kalman_pair, rolling_pair
from tradebot.analysis.reversion import check_filter, reversion
from tradebot.backtest.pairs import next_side
from tradebot.brokers.base import BaseBroker
from tradebot.core.models import Order, OrderStatus, Side
from tradebot.core.timeframes import to_timedelta
from tradebot.journal import Journal
from tradebot.live.risk import check_stake

log = logging.getLogger(__name__)
STATE_VERSION = 1
EPS = 1e-12
FULL = 1 - 1e-3  #: a fill this close to the order counts as complete (exchange rounding)
TOLERANCE = 0.01  #: the account may hold this much less than the saved pair (fees taken in the asset)


class PairStateError(RuntimeError):
    """The pair bot can't trust what it holds (state file unreadable, a trade interrupted, broker disagrees):
    it refuses to trade until the account has been checked."""


def pair_state_path(broker: str, legs: tuple[str, str], sandbox: bool = False) -> Path:
    """Where ``bot pair-run --live`` keeps the open pair: one file per broker (testnet apart) and pair, in
    ``$TRADEBOT_PAIR_STATE_DIR`` or the ``pair_bots`` folder of the OS config dir."""
    from tradebot.config import os_config_dir

    def safe(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")

    folder = os.environ.get("TRADEBOT_PAIR_STATE_DIR")
    return (Path(folder).expanduser() if folder else os_config_dir() / "pair_bots") /f"{safe(broker)}{'-testnet' if sandbox else ''}_{safe(legs[0])}_{safe(legs[1])}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, data: dict) -> None:
    """Temp file in the same folder, fsync, then rename: a crash leaves the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_pair_state(path: Path | str) -> dict | None:
    """The saved state, ``None`` without a file; :class:`PairStateError` when it is unreadable or malformed."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("version") != STATE_VERSION:
            raise ValueError(f"unknown version {s.get('version')!r}")
        legs, amounts = [str(x) for x in s["legs"]], [float(x) for x in s["amounts"]]
        side = float(s["side"])
        if len(legs) != 2 or len(amounts) != 2 or side not in (-1.0, 0.0, 1.0) or min(amounts) < 0:
            raise ValueError("bad legs, side or amounts")
        if s.get("pending") is not None and not isinstance(s["pending"], dict):
            raise ValueError("bad pending trade")
        s["legs"], s["amounts"], s["side"], s["closing"] = legs, amounts, side, bool(s.get("closing"))
        return s
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PairStateError(f"pair state file {path} is unreadable ({exc}); check the positions at the broker, "
                             "then fix the file or delete it") from exc


def describe_pending(s: dict) -> str:
    p = s["pending"]
    orders = "; ".join(
        f"{o['side']} {o['amount']:.8g} {o['symbol']} "
        + ("(may have been placed)" if o.get("filled") is None else f"(filled {o['filled']:.8g})")
        for o in p.get("orders", [])) or "none"
    problem = f": {p['problem']}" if p.get("problem") else ""
    return f"the trade '{p.get('action')}' started {p.get('started')} did not finish{problem}. Orders: {orders}"


def _ambiguous(exc: BaseException) -> bool:
    """A failure after which the order may still have reached the exchange (timeouts, lost connections)."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        import ccxt
    except ImportError:
        return False
    return isinstance(exc, ccxt.NetworkError)


@dataclass
class PairEvent:
    time: datetime
    pair: str
    z: float
    beta: float
    action: str  #: "open long", "open short", "close" or "hold" (long = long A / short B)
    price_a: float
    price_b: float
    amount_a: float = 0.0
    amount_b: float = 0.0
    orders: list[Order] = field(default_factory=list)
    dry_run: bool = True
    note: str = ""
    #: chance that a trade opened now closes before the stop within the horizon (#223); NaN when not computed
    probability: float = float("nan")


@dataclass
class PairBot:
    broker: BaseBroker
    legs: tuple[str, str]
    timeframe: str = "1h"
    stake: float = 100.0  #: quote-currency value of both legs together per entry
    window: int = 60
    entry: float = 2.0
    exit: float = 0.5
    stop: float = 4.0
    method: str = "rolling"
    history: int = 500
    dry_run: bool = True
    on_event: Callable[[PairEvent], None] | None = None
    side: float = 0.0  #: +1 long A / short B, -1 the reverse, 0 flat
    amounts: tuple[float, float] = (0.0, 0.0)  #: what the open pair holds in each leg
    events: list[PairEvent] = field(default_factory=list)
    state_path: Path | str | None = None  #: keep the open pair here across restarts (live only)
    fill_timeout: float = 30.0  #: seconds to wait for an open order before cancelling the rest
    poll_seconds: float = 1.0
    closing: bool = False  #: a close that didn't finish: the rest is closed at the next candle
    opened: str | None = None  #: when the open pair was opened (ISO time)
    halted: str = ""  #: why the bot stopped trading (an outcome it can't know); a restart needs a check
    journal: Journal | None = None  #: every order is stored here (#159)
    #: open only when the gap's chance to close before the stop within ``horizon`` candles is this high (#223); 0 = off
    min_probability: float = 0.0
    horizon: float = 30

    def __post_init__(self):
        self.stake = check_stake(self.stake)
        if not 0 <= self.exit < self.entry < self.stop:
            raise ValueError("need 0 <= exit < entry < stop")
        self.min_probability, self.horizon = check_filter(self.min_probability, self.horizon)
        if self.method not in ("rolling", "kalman"):
            raise ValueError("method must be rolling or kalman")
        self.legs = (str(self.legs[0]), str(self.legs[1]))
        if self.state_path is not None:
            if self.dry_run:
                raise ValueError("the pair state is only kept for live trading")
            self.state_path = Path(self.state_path)
            self._resume()

    @property
    def name(self) -> str:
        return f"{self.legs[0]}:{self.legs[1]}"

    # --- state -------------------------------------------------------------------------------------------
    def _resume(self) -> None:
        s = read_pair_state(self.state_path)
        if s is None:
            return
        if s["legs"] != list(self.legs):
            raise PairStateError(f"{self.state_path} belongs to {s['legs'][0]}:{s['legs'][1]}, not {self.name}")
        if s.get("pending"):
            raise PairStateError(
                f"{self.name}: {describe_pending(s)}. Check the positions at the broker, then run "
                "`tradebot bot pair-state` with --keep (the saved pair is right) or --forget (these legs are flat)")
        self.side, self.amounts, self.closing = s["side"], tuple(s["amounts"]), s["closing"]
        self.opened = s.get("opened")
        if self.side:
            log.info("%s: resumed the open pair %s (%.8g / %.8g) from %s", self.name,
                     "long" if self.side > 0 else "short", *self.amounts, self.state_path)

    def state(self, pending: dict | None = None) -> dict:
        return {"version": STATE_VERSION, "legs": list(self.legs), "side": self.side, "amounts": list(self.amounts),
                "closing": self.closing, "opened": self.opened, "updated": _now(), "pending": pending}

    def _save(self, pending: dict | None = None) -> None:
        if self.state_path is not None:
            write_json_atomic(self.state_path, self.state(pending))

    def _begin(self, action: str) -> dict:
        pending = {"action": action, "started": _now(), "orders": []}
        self._save(pending)  # written ahead: a crash from here on is seen at the next start
        return pending

    def _halt(self, pending: dict, problem: str) -> PairStateError:
        pending["problem"] = problem
        try:
            self._save(pending)
        except OSError as exc:
            log.critical("%s: can't write the pair state (%s)", self.name, exc)
        return PairStateError(f"{self.name}: {problem}")

    def expected(self) -> tuple[float, float]:
        """Signed amount the open pair should hold in each leg (0, 0 when flat)."""
        if not self.side:
            return 0.0, 0.0
        s = 1.0 if self.side > 0 else -1.0
        return s * self.amounts[0], -s * self.amounts[1]

    def reconcile(self) -> list[str]:
        """Check the saved pair against the broker's positions. Raises :class:`PairStateError` when the
        account holds less than the pair (or the other direction); returns warnings that don't stop it."""
        if self.dry_run:
            return []
        problems, notes = [], []
        for leg, want in zip(self.legs, self.expected()):
            held = self.broker.leg_position(leg)
            if held is None:
                if want:
                    notes.append(f"{leg}: the broker can't report positions; trusting the saved pair")
            elif not want:
                if abs(held) > EPS:
                    notes.append(f"{leg}: the account holds {held:+.8g} although this bot has no open pair")
            elif held * want <= 0 or abs(held) < abs(want) * (1 - TOLERANCE):
                problems.append(f"{leg}: the saved pair holds {want:+.8g}, the account {held:+.8g}")
        for n in notes:
            log.warning("%s: %s", self.name, n)
        if problems:
            raise PairStateError(f"{self.name}: the account doesn't match the saved pair ({'; '.join(problems)}). "
                                 "Fix the positions or run `tradebot bot pair-state` with --forget")
        return notes

    # --- orders ------------------------------------------------------------------------------------------
    def _fill(self, order: Order, requested: float) -> float:
        """How much of ``order`` filled: waits for an open order, cancels the rest after ``fill_timeout``."""
        o, known = order, True
        deadline = time.monotonic() + self.fill_timeout
        while o.status is OrderStatus.OPEN and known:
            if time.monotonic() >= deadline:
                log.warning("%s: order %s (%s) still open after %ss; cancelling the rest", self.name, o.id,
                            o.symbol, self.fill_timeout)
                try:
                    self.broker.cancel_order(o.id, o.symbol)
                except Exception as exc:
                    log.warning("%s: cancelling %s failed (%s)", self.name, o.id, exc)
                o = self.broker.get_order(o.id, o.symbol) or o
                if o.status is OrderStatus.OPEN:
                    raise PairStateError(f"order {o.id} ({o.symbol}) is still open and couldn't be cancelled")
                break
            if self.poll_seconds:
                time.sleep(self.poll_seconds)
            got = self.broker.get_order(o.id, o.symbol)
            known, o = got is not None, got or o
        if not known:  # still open and the broker can't tell how much filled: no guessing with real money
            raise PairStateError(f"order {o.id} ({o.symbol}) is open and the broker can't look it up")
        if o.status in (OrderStatus.CLOSED, OrderStatus.PENDING_MANUAL):
            return o.filled if o.filled > 0 else (o.amount or requested)
        return o.filled  # cancelled (or the rest cancelled): what filled

    def _journal(self, order: Order, note: str) -> None:
        if self.journal:
            try:
                mode = "paper" if getattr(self.broker, "simulated", False) else "live"
                self.journal.record_order(order, self.broker, mode=mode, bot=f"pair {self.name}", note=note)
            except Exception:  # a broken journal must never stop the bot
                log.exception("journal write failed for %s", order.symbol)

    def _place(self, symbol: str, side: Side, amount: float, pending: dict, note: str = "") -> tuple[Order, float]:
        rec = {"symbol": symbol, "side": side.value, "amount": amount, "id": None, "filled": None}
        pending["orders"].append(rec)
        self._save(pending)
        try:
            order = self.broker.place_order(symbol, side, amount)
        except Exception as exc:
            if _ambiguous(exc):
                raise self._halt(pending, f"{side.value} {amount:.8g} {symbol} failed with {exc!r}; "
                                          "the order may have gone through") from exc
            rec["filled"], rec["error"] = 0.0, str(exc)
            self._save(pending)
            raise
        rec["id"] = order.id
        try:
            filled = self._fill(order, amount)
        except Exception as exc:
            self._journal(order, (note + " " if note else "") + "fill unknown")
            raise self._halt(pending, f"can't tell how much of order {order.id} ({side.value} {symbol}) filled: "
                                      f"{exc}") from exc
        rec["filled"] = filled
        if filled > 0:
            self._journal(dataclasses.replace(order, filled=filled), note)
        try:
            self._save(pending)
        except OSError as exc:
            raise self._halt(pending, f"can't write the pair state: {exc}") from exc
        return order, filled

    def _open(self, new: float, want: tuple[float, float], event: PairEvent) -> None:
        buy_leg = 0 if new > 0 else 1
        buy, sell = self.legs[buy_leg], self.legs[1 - buy_leg]
        pending = self._begin(event.action)
        try:
            first, f1 = self._place(buy, Side.BUY, want[buy_leg], pending)
        except PairStateError:
            raise
        except Exception:
            self._save()  # rejected: nothing was bought, still flat
            raise
        event.orders.append(first)
        if f1 <= EPS:
            self._save()
            event.action, event.amount_a, event.amount_b = "hold", 0.0, 0.0
            event.note = f"buying {buy} didn't fill; nothing opened"
            return
        sell_amt = want[1 - buy_leg] * min(1.0, f1 / want[buy_leg])  # a partial first leg shrinks the second
        try:
            second, f2 = self._place(sell, Side.SELL, sell_amt, pending)
            event.orders.append(second)
            if f2 <= EPS:
                raise RuntimeError(f"selling {sell} didn't fill")
        except PairStateError:
            raise
        except Exception as exc:
            log.error("%s: selling %s failed (%s), undoing the purchase of %.8g %s", self.name, sell, exc, f1, buy)
            try:
                undo, fu = self._place(buy, Side.SELL, f1, pending, "undo")
            except PairStateError:
                raise
            except Exception as exc2:
                raise self._halt(pending, f"selling {sell} failed ({exc}) and undoing the purchase failed too "
                                          f"({exc2}): the account holds {f1:.8g} {buy}") from exc2
            event.orders.append(undo)
            if fu < f1 * FULL:
                raise self._halt(pending, f"selling {sell} failed ({exc}) and the undo sold only {fu:.8g} of "
                                          f"{f1:.8g} {buy}: the account holds the rest") from exc
            self._save()  # back to flat
            raise
        fills = [0.0, 0.0]
        fills[buy_leg], fills[1 - buy_leg] = f1, f2
        if f1 < want[buy_leg] * FULL or f2 < sell_amt * FULL:
            event.note = "partly filled"
            if f2 < sell_amt * FULL:
                event.note += ": the legs are not in the hedge ratio"
        self.side, self.amounts, self.closing, self.opened = new, (fills[0], fills[1]), False, _now()
        event.amount_a, event.amount_b = fills
        self._save()

    def _close(self, event: PairEvent) -> None:
        left = list(self.amounts)
        buy_leg = 1 if self.side > 0 else 0  # long gap = long A / short B, so closing buys B back
        pending = self._begin("close")
        error = None
        for leg, side in ((buy_leg, Side.BUY), (1 - buy_leg, Side.SELL)):
            if left[leg] <= EPS:
                continue
            try:
                order, f = self._place(self.legs[leg], side, left[leg], pending)
            except PairStateError:
                raise
            except Exception as exc:
                error = exc
                break  # the other leg stays open as well: still hedged when nothing was closed
            event.orders.append(order)
            left[leg] = 0.0 if f >= left[leg] * FULL else left[leg] - f
        if max(left) <= EPS:
            self.side, self.amounts, self.closing, self.opened = 0.0, (0.0, 0.0), False, None
        else:
            changed = tuple(left) != tuple(self.amounts)
            self.amounts, self.closing = (left[0], left[1]), self.closing or changed
            event.note = (f"closed partly; {left[0]:.8g} {self.legs[0]} / {left[1]:.8g} {self.legs[1]} left, "
                          "closed at the next candle") if self.closing else "close failed; the pair stays open"
        self._save()
        if error is not None:
            log.error("%s: closing failed (%s); %s", self.name, error, event.note)
            raise error

    # --- loop --------------------------------------------------------------------------------------------
    def closed_closes(self) -> tuple[pd.Series, pd.Series]:
        """Close prices of both legs (the second in the first's currency), without the candle still forming."""
        a, b = self.broker.data.pair_legs(self.legs, self.timeframe, bars=self.history)
        ca, cb = a["close"], b["close"]
        step = to_timedelta(self.timeframe)
        if len(ca) and ca.index[-1] + step > pd.Timestamp.now(tz="UTC"):
            ca, cb = ca.iloc[:-1], cb.iloc[:-1]
        both = pd.concat([ca, cb], axis=1, join="inner").dropna()
        return both.iloc[:, 0], both.iloc[:, 1]

    def step(self) -> PairEvent:
        if self.halted:
            raise PairStateError(f"{self.name} halted: {self.halted}")
        ca, cb = self.closed_closes()
        if len(ca) < self.window + 2:
            raise ValueError(f"need at least {self.window + 2} common candles, got {len(ca)}")
        roll = kalman_pair(ca, cb) if self.method == "kalman" else rolling_pair(ca, cb, self.window)
        z, beta = float(roll.zscore.iloc[-1]), float(roll.beta.iloc[-1])
        pa, pb = float(ca.iloc[-1]), float(cb.iloc[-1])
        prob = float(reversion(roll.zscore, self.horizon, self.exit, self.stop, 2 * self.window)["probability"].iloc[-1])
        new = 0.0 if self.closing else next_side(self.side, z, beta, self.entry, self.exit, self.stop, prob,
                                                 self.min_probability)
        event = PairEvent(datetime.now(timezone.utc), self.name, z, beta, "hold", pa, pb, dry_run=self.dry_run,
                          probability=prob)
        if new == 0.0 == self.side and not self.closing and self.min_probability > 0 \
                and next_side(0.0, z, beta, self.entry, self.exit, self.stop) != 0.0:  # only the filter said no
            event.note = (f"not opened: the gap closes before the stop with {prob:.1%} chance in {self.horizon:g} "
                          f"candles, below {self.min_probability:.1%}" if prob == prob else
                          "not opened: the chance to close is unknown (the gap shows no reversion over the last "
                          f"{2 * self.window} candles, or there are fewer than {3 * self.window + 2})")
        if new != self.side:
            try:
                if self.side != 0.0:  # close what is open: the reverse of the entry, same amounts
                    event.action, (event.amount_a, event.amount_b) = "close", self.amounts
                    if self.dry_run:
                        self.side, self.amounts = 0.0, (0.0, 0.0)
                    else:
                        self._close(event)
                else:
                    want = (self.stake / (1 + beta) / pa, self.stake * beta / (1 + beta) / pb)
                    event.action = "open long" if new > 0 else "open short"
                    event.amount_a, event.amount_b = want
                    if self.dry_run:
                        self.side, self.amounts = new, want
                    else:
                        self._open(new, want, event)
            except PairStateError as exc:
                self.halted = str(exc)
                log.critical("%s", exc)
                notify.notify("error", f"Pair bot halted: {self.name}", str(exc))
                raise
        self.events.append(event)
        log.info("%s %s z=%+.2f β=%.2f%s%s", self.name, event.action, z, beta, " (dry run)" if self.dry_run else "",
                 f" [{event.note}]" if event.note else "")
        if self.on_event:
            self.on_event(event)
        return event

    def run(self, max_steps: int | None = None) -> None:
        """Evaluate once per candle close until interrupted (or ``max_steps`` reached). A halt (an outcome the
        bot can't know) ends the loop with :class:`PairStateError`."""
        self.reconcile()
        step = to_timedelta(self.timeframe).total_seconds()
        n = 0
        while max_steps is None or n < max_steps:
            try:
                self.step()
            except PairStateError:
                raise
            except Exception:
                log.exception("pair bot step failed for %s; retrying next candle", self.name)
            n += 1
            if max_steps is not None and n >= max_steps:
                break
            time.sleep(step - time.time() % step + 2)  # wake just after the next candle closes
