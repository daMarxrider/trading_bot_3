# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Live trading loop.

Uses the exact execution model of the backtester: on each *closed* candle the
algorithm's target position is evaluated and the bot buys/sells to match it.
``stake`` caps the quote-currency value of a single entry (the legacy
``max_trading_value``). Dry-run is the default; nothing is sent unless
``dry_run=False``.

With a sizing model (``tradebot.algorithms.sizing``, #147) the entry buys ``stake`` times the size instead of
the whole stake; the target it follows is the algorithm's own signal.

A target above 0.5 buys; once in, a lower target sells that share of the holding (a combo's take-profits in
steps scale out, #150) and a target of 0 sells everything. The bot follows the same rules as the backtest
(the algorithm decides on the closed candles), but acts at the close: a combo's intrabar stop that the
backtest fills inside a candle is sold here at that candle's close.

With a ``stream`` (:class:`tradebot.data.stream.MarketStream`, #158) the bot also watches the price between the
closes. A combo with ``fills="intrabar"`` then sells inside the candle when the candle's low reaches its stop
or its high a take-profit (the levels the backtest fills against, :meth:`~tradebot.algorithms.combo.Combo.
exit_levels`), and the risk guard checks the open profit on every price, so a halt with ``close_on_halt``
sells right away. Entries (a limit entry too) still go in at the close. Without the stream, or when it falls
back to polling, the same happens on every poll.
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from tradebot import notify
from tradebot.algorithms.base import BaseAlgorithm
from tradebot.brokers.base import BaseBroker
from tradebot.core.models import Order, Side
from tradebot.core.timeframes import to_timedelta
from tradebot.journal import Journal
from tradebot.live.risk import RiskEvent, RiskGuard, RiskLimits

log = logging.getLogger(__name__)


@dataclass
class BotEvent:
    time: datetime
    symbol: str
    target: float
    action: str  #: "buy", "sell" or "hold"
    price: float
    amount: float = 0.0
    order: Order | None = None
    dry_run: bool = True
    note: str = ""


@dataclass
class TradingBot:
    broker: BaseBroker
    algorithm: BaseAlgorithm
    symbol: str
    timeframe: str = "1h"
    stake: float = 100.0
    history: int = 500
    dry_run: bool = True
    on_event: Callable[[BotEvent], None] | None = None
    holding: float = 0.0  #: base-asset amount this bot currently manages
    fraction: float = 0.0  #: the target the holding stands for (1 after a full entry, less after scaling out)
    events: list[BotEvent] = field(default_factory=list)
    risk: RiskLimits | None = None  #: safety limits (#154); see :mod:`tradebot.live.risk`
    on_risk: Callable[[RiskEvent], None] | None = None  #: called on every risk trigger (notification hook)
    entry_price: float = 0.0  #: price the current position was bought at
    guard: RiskGuard | None = field(default=None, init=False, repr=False)
    journal: Journal | None = None  #: every order is stored here (#159); see :mod:`tradebot.journal`
    stream: object | None = None  #: a :class:`~tradebot.data.stream.MarketStream`: act inside the candle (#158)
    levels: dict | None = field(default=None, init=False, repr=False)  #: the intrabar exits for the current candle
    _levels_for: pd.Timestamp | None = field(default=None, init=False, repr=False)  #: the candle they apply to

    def _risk_trigger(self, e: RiskEvent) -> None:
        notify.notify("risk", f"Risk guard: {self.symbol} {e.kind}", e.message)
        if self.on_risk:
            self.on_risk(e)

    def __post_init__(self):
        if self.risk is not None:
            self.guard = RiskGuard(self.risk, self.stake, self._risk_trigger)
        if getattr(self.broker, "simulated", False) and not self.dry_run:  # resume a persistent paper account
            held = self.broker.get_balance().get(self.base_asset)
            self.holding = held.total if held else 0.0

    @property
    def base_asset(self) -> str:
        return self.symbol.split("/")[0]

    def closed_candles(self) -> pd.DataFrame:
        df = self.broker.get_klines(self.symbol, self.timeframe, self.history)
        step = to_timedelta(self.timeframe)
        if len(df) and df.index[-1] + step > pd.Timestamp.now(tz="UTC"):
            df = df.iloc[:-1]  # the last candle is still forming
        return df

    @property
    def mode(self) -> str:
        return "dry" if self.dry_run else "paper" if getattr(self.broker, "simulated", False) else "live"

    def _journal(self, event: BotEvent) -> None:
        if not self.journal or event.action not in ("buy", "sell"):
            return
        try:
            bot = f"{self.algorithm.__class__.__name__} {self.timeframe}"
            if event.order:
                self.journal.record_order(event.order, self.broker, mode=self.mode, bot=bot, note=event.note)
            else:  # dry run: nothing was sent, record the intended fill at the candle's close
                self.journal.record(self.symbol, event.action, event.amount, event.price, mode=self.mode,
                                    broker=getattr(self.broker, "name", ""), bot=bot, note=event.note)
        except Exception:  # a broken journal must never stop the bot
            log.exception("journal write failed for %s", self.symbol)

    def _sell(self, event: BotEvent, target: float = 0.0) -> None:
        """Sell the whole holding (``target`` 0) or scale out to ``target`` of the position."""
        available = self.broker.get_balance().get(self.base_asset) if not self.dry_run else None
        fraction = self.fraction if self.fraction > 0 else 1.0
        event.action = "sell"
        share = self.holding if target <= 1e-9 else self.holding * (1 - target / fraction)
        event.amount = min(share, available.free) if available else share
        if target > 1e-9:
            event.note = f"scale out to {target:.0%} of the position"
        if not self.dry_run:
            event.order = self.broker.place_order(self.symbol, Side.SELL, event.amount)
        fill = (event.order.price if event.order else None) or event.price
        if target <= 1e-9:
            self.holding, self.fraction = 0.0, 0.0
        else:
            self.holding, self.fraction = self.holding - event.amount, target
        if self.guard:
            self.guard.record_trade((fill - self.entry_price) * event.amount)
            self.guard.check()  # a loss that breaks a limit halts right away

    def step(self) -> BotEvent:
        df = self.closed_candles()
        size = 1.0
        if hasattr(self.algorithm, "run_sized"):  # position sizing (#147): follow the signal, size the entry
            signal, sizes = self.algorithm.run_sized(df)
            target, size = float(signal.iloc[-1]), float(sizes.iloc[-1])
        else:
            target = float(self.algorithm.run(df).iloc[-1])
        price = float(df["close"].iloc[-1])
        event = BotEvent(datetime.now(timezone.utc), self.symbol, target, "hold", price, dry_run=self.dry_run)
        fraction = self.fraction if self.fraction > 0 else 1.0
        halted = self.guard.check(self.holding * (price - self.entry_price) if self.holding > 0 else 0.0) \
            if self.guard else None
        if halted:
            event.note = f"halted: {halted.message}"
            if self.risk.close_on_halt and self.holding > 0:
                self._sell(event)
        elif target > 0.5 and self.holding <= 0:
            event.action, event.amount = "buy", self.stake * size / price
            if size < 1:
                event.note = f"sized to {size:.0%} of the stake"
            if self.guard:
                market = self.broker.get_ticker(self.symbol).last if self.risk.price_band else None
                veto = self.guard.check_entry(event.amount, price, self.holding, market)
                if veto:
                    event.action, event.amount, event.note = "hold", 0.0, f"rejected: {veto.message}"
            if event.action == "buy":
                if not self.dry_run:
                    event.order = self.broker.place_order(self.symbol, Side.BUY, event.amount)
                self.holding, self.fraction = event.amount, target
                self.entry_price = (event.order.price if event.order else None) or price
        elif self.holding > 0 and target < fraction - 1e-9:
            self._sell(event, target)
        if self.stream is not None:
            self.levels = exit_levels(self.algorithm, df) if self.holding > 0 else None
            self._levels_for = df.index[-1] + to_timedelta(self.timeframe) if len(df) else None
        return self._finish(event)

    def _finish(self, event: BotEvent) -> BotEvent:
        self.events.append(event)
        self._journal(event)
        log.info("%s %s target=%.2f price=%.6g amount=%.6g%s%s", self.symbol, event.action, event.target,
                 event.price, event.amount, " (dry run)" if self.dry_run else "",
                 f" [{event.note}]" if event.note else "")
        if self.on_event:
            self.on_event(event)
        return event

    def on_price(self, price: float, low: float | None = None, high: float | None = None) -> BotEvent | None:
        """A price between two closes (#158): sell on a stop, a take-profit or a risk halt inside the candle.

        ``low``/``high``: the extremes of the candle so far (from the stream's forming candle), so a level the
        price went through between two updates counts too. Returns the event when the bot sold, else None."""
        if self.holding <= 0:
            return None
        low = price if low is None else min(low, price)
        high = price if high is None else max(high, price)
        event = BotEvent(datetime.now(timezone.utc), self.symbol, self.fraction, "hold", price, dry_run=self.dry_run)
        if self.guard:
            halted = self.guard.check(self.holding * (price - self.entry_price))
            if halted:
                if not self.risk.close_on_halt:
                    return None
                event.target = 0.0
                self._sell(event)
                event.note = f"halted inside the candle: {halted.message}"
                self.levels = None
                return self._finish(event)
        lv = self.levels
        if not lv:
            return None
        stop = lv.get("stop")
        if stop is not None and low <= stop:
            event.target = 0.0
            self._sell(event)
            event.note = f"stop {stop:.6g} reached inside the candle"
            self.levels = None
            return self._finish(event)
        hit = [(level, size) for level, size in lv.get("take_profits", []) if high >= level]
        if not hit:
            return None
        lv["take_profits"] = [t for t in lv["take_profits"] if t not in hit]
        fraction = self.fraction if self.fraction > 0 else 1.0
        target = max(fraction - sum(size for _, size in hit), 0.0)
        event.target = target
        self._sell(event, target)
        event.note = f"take-profit {max(level for level, _ in hit):.6g} reached inside the candle" + (
            f", {event.note}" if event.note else "")
        if target <= 1e-9:
            self.levels = None
        return self._finish(event)

    def _on_tick(self, tick) -> None:
        current = self._levels_for is not None and tick.candle[0] == self._levels_for
        low, high = (tick.candle[3], tick.candle[2]) if current else (None, None)  # a later candle: its price only
        try:
            ev = self.on_price(tick.price, low, high)
            if ev is not None:
                notify.bot_event(ev)
        except Exception:
            log.exception("intrabar check failed for %s", self.symbol)

    def _wait(self, until: float, ticks: "queue.Queue | None") -> None:
        """Sleep until ``until`` (epoch seconds), handling the stream's prices meanwhile."""
        while (left := until - time.time()) > 0:
            if ticks is None:
                time.sleep(left)
                return
            try:
                tick = ticks.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            self._on_tick(tick)

    def run(self, max_steps: int | None = None) -> None:
        """Evaluate once per candle close until interrupted (or ``max_steps`` reached)."""
        step = to_timedelta(self.timeframe).total_seconds()
        ticks, started = None, False
        if self.stream is not None and max_steps != 1:
            ticks = queue.Queue()
            self.stream.on_tick(ticks.put)
            if not self.stream.running:
                self.stream.start()
                started = True
        n = 0
        try:
            while max_steps is None or n < max_steps:
                try:
                    notify.bot_event(self.step())
                except Exception:
                    log.exception("bot step failed for %s; retrying next candle", self.symbol)
                    notify.notify("error", f"Bot step failed: {self.symbol}", "See the log; retrying next candle.")
                n += 1
                if max_steps is not None and n >= max_steps:
                    break
                now = time.time()
                self._wait(now + step - now % step + 2, ticks)  # wake just after the next candle closes
        finally:
            if ticks is not None:
                self.stream.off_tick(ticks.put)
                if started:
                    self.stream.stop()


def exit_levels(algorithm, df: pd.DataFrame) -> dict | None:
    """The algorithm's intrabar exits for the next candle (a combo with ``fills="intrabar"``), else None."""
    inner = algorithm.__dict__.get("inner")  # a sized algorithm (#147): the combo inside, with its parameters
    try:
        if inner is not None and hasattr(inner, "exit_levels"):
            return inner.exit_levels(df, algorithm._inner_params(algorithm.values))
        if hasattr(type(algorithm), "exit_levels"):
            return algorithm.exit_levels(df)
    except Exception:
        log.exception("couldn't work out the intrabar exits")
    return None
