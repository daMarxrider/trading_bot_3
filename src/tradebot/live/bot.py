# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Live trading loop.

Uses the exact execution model of the backtester: on each *closed* candle the
algorithm's target position is evaluated and the bot buys/sells to match it.
``stake`` caps the quote-currency value of a single entry (the legacy
``max_trading_value``). Dry-run is the default; nothing is sent unless
``dry_run=False``.
"""

from __future__ import annotations

import logging
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
    events: list[BotEvent] = field(default_factory=list)
    risk: RiskLimits | None = None  #: safety limits (#154); see :mod:`tradebot.live.risk`
    on_risk: Callable[[RiskEvent], None] | None = None  #: called on every risk trigger (notification hook)
    entry_price: float = 0.0  #: price the current position was bought at
    guard: RiskGuard | None = field(default=None, init=False, repr=False)

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

    def _sell(self, event: BotEvent) -> None:
        available = self.broker.get_balance().get(self.base_asset) if not self.dry_run else None
        event.action = "sell"
        event.amount = min(self.holding, available.free) if available else self.holding
        if not self.dry_run:
            event.order = self.broker.place_order(self.symbol, Side.SELL, event.amount)
        fill = (event.order.price if event.order else None) or event.price
        self.holding = 0.0
        if self.guard:
            self.guard.record_trade((fill - self.entry_price) * event.amount)
            self.guard.check()  # a loss that breaks a limit halts right away

    def step(self) -> BotEvent:
        df = self.closed_candles()
        target = float(self.algorithm.run(df).iloc[-1])
        price = float(df["close"].iloc[-1])
        event = BotEvent(datetime.now(timezone.utc), self.symbol, target, "hold", price, dry_run=self.dry_run)
        halted = self.guard.check(self.holding * (price - self.entry_price) if self.holding > 0 else 0.0) \
            if self.guard else None
        if halted:
            event.note = f"halted: {halted.message}"
            if self.risk.close_on_halt and self.holding > 0:
                self._sell(event)
        elif target > 0.5 and self.holding <= 0:
            event.action, event.amount = "buy", self.stake / price
            if self.guard:
                market = self.broker.get_ticker(self.symbol).last if self.risk.price_band else None
                veto = self.guard.check_entry(event.amount, price, self.holding, market)
                if veto:
                    event.action, event.amount, event.note = "hold", 0.0, f"rejected: {veto.message}"
            if event.action == "buy":
                if not self.dry_run:
                    event.order = self.broker.place_order(self.symbol, Side.BUY, event.amount)
                self.holding, self.entry_price = event.amount, (event.order.price if event.order else None) or price
        elif target <= 0.5 and self.holding > 0:
            self._sell(event)
        self.events.append(event)
        log.info("%s %s target=%.2f price=%.6g amount=%.6g%s%s", self.symbol, event.action, target, price,
                 event.amount, " (dry run)" if self.dry_run else "", f" [{event.note}]" if event.note else "")
        if self.on_event:
            self.on_event(event)
        return event

    def run(self, max_steps: int | None = None) -> None:
        """Evaluate once per candle close until interrupted (or ``max_steps`` reached)."""
        step = to_timedelta(self.timeframe).total_seconds()
        n = 0
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
            time.sleep(step - now % step + 2)  # wake just after the next candle closes
