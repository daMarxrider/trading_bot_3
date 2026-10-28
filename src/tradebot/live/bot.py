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

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.brokers.base import BaseBroker
from tradebot.core.models import Order, Side
from tradebot.core.timeframes import to_timedelta

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

    @property
    def base_asset(self) -> str:
        return self.symbol.split("/")[0]

    def closed_candles(self) -> pd.DataFrame:
        df = self.broker.get_klines(self.symbol, self.timeframe, self.history)
        step = to_timedelta(self.timeframe)
        if len(df) and df.index[-1] + step > pd.Timestamp.now(tz="UTC"):
            df = df.iloc[:-1]  # the last candle is still forming
        return df

    def step(self) -> BotEvent:
        df = self.closed_candles()
        target = float(self.algorithm.run(df).iloc[-1])
        price = float(df["close"].iloc[-1])
        event = BotEvent(datetime.now(timezone.utc), self.symbol, target, "hold", price, dry_run=self.dry_run)
        if target > 0.5 and self.holding <= 0:
            event.action, event.amount = "buy", self.stake / price
        elif target <= 0.5 and self.holding > 0:
            available = self.broker.get_balance().get(self.base_asset) if not self.dry_run else None
            event.action = "sell"
            event.amount = min(self.holding, available.free) if available else self.holding
        if event.action != "hold":
            if not self.dry_run:
                event.order = self.broker.place_order(self.symbol, Side(event.action), event.amount)
            self.holding = event.amount if event.action == "buy" else 0.0
        self.events.append(event)
        log.info("%s %s target=%.2f price=%.6g amount=%.6g%s", self.symbol, event.action, target, price,
                 event.amount, " (dry run)" if self.dry_run else "")
        if self.on_event:
            self.on_event(event)
        return event

    def run(self, max_steps: int | None = None) -> None:
        """Evaluate once per candle close until interrupted (or ``max_steps`` reached)."""
        step = to_timedelta(self.timeframe).total_seconds()
        n = 0
        while max_steps is None or n < max_steps:
            try:
                self.step()
            except Exception:
                log.exception("bot step failed for %s; retrying next candle", self.symbol)
            n += 1
            if max_steps is not None and n >= max_steps:
                break
            now = time.time()
            time.sleep(step - now % step + 2)  # wake just after the next candle closes
