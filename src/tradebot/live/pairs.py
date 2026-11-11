# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Live two-legged pairs trading (#117): long one asset and short the other at the same time.

Same rule as :func:`tradebot.backtest.pairs.pair_backtest` (:func:`~tradebot.backtest.pairs.next_side`), on
each *closed* candle. The stake is split by the hedge ratio at entry, ``stake / (1 + β)`` in leg A and
``stake · β / (1 + β)`` in leg B, and the two amounts are kept until the exit. Dry-run is the default.

The short leg is an ordinary SELL, so the broker needs margin or futures for it: pass the legs as the symbols
that broker can short (e.g. perpetual contracts ``BTC/USDT:USDT``). A spot account rejects the sale; the bot
then undoes the leg already placed and stops trading that step. It doesn't remember an open pair across a
restart (like :class:`~tradebot.live.bot.TradingBot`): check the broker's positions before starting again.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from tradebot.analysis.pairs import kalman_pair, rolling_pair
from tradebot.backtest.pairs import next_side
from tradebot.brokers.base import BaseBroker
from tradebot.core.models import Order, Side
from tradebot.core.timeframes import to_timedelta

log = logging.getLogger(__name__)


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

    def __post_init__(self):
        if not 0 <= self.exit < self.entry < self.stop:
            raise ValueError("need 0 <= exit < entry < stop")
        if self.method not in ("rolling", "kalman"):
            raise ValueError("method must be rolling or kalman")

    @property
    def name(self) -> str:
        return f"{self.legs[0]}:{self.legs[1]}"

    def closed_closes(self) -> tuple[pd.Series, pd.Series]:
        """Close prices of both legs (the second in the first's currency), without the candle still forming."""
        a, b = self.broker.data.pair_legs(self.legs, self.timeframe, bars=self.history)
        ca, cb = a["close"], b["close"]
        step = to_timedelta(self.timeframe)
        if len(ca) and ca.index[-1] + step > pd.Timestamp.now(tz="UTC"):
            ca, cb = ca.iloc[:-1], cb.iloc[:-1]
        both = pd.concat([ca, cb], axis=1, join="inner").dropna()
        return both.iloc[:, 0], both.iloc[:, 1]

    def _trade(self, buy_leg: int, amount_buy: float, amount_sell: float) -> list[Order]:
        """Buy leg ``buy_leg`` and sell the other. If the second order fails the first is undone."""
        buy, sell = self.legs[buy_leg], self.legs[1 - buy_leg]
        first = self.broker.place_order(buy, Side.BUY, amount_buy)
        try:
            second = self.broker.place_order(sell, Side.SELL, amount_sell)
        except Exception:
            log.error("%s: selling %s failed, undoing the purchase of %s", self.name, sell, buy)
            self.broker.place_order(buy, Side.SELL, amount_buy)
            raise
        return [first, second]

    def step(self) -> PairEvent:
        ca, cb = self.closed_closes()
        if len(ca) < self.window + 2:
            raise ValueError(f"need at least {self.window + 2} common candles, got {len(ca)}")
        roll = kalman_pair(ca, cb) if self.method == "kalman" else rolling_pair(ca, cb, self.window)
        z, beta = float(roll.zscore.iloc[-1]), float(roll.beta.iloc[-1])
        pa, pb = float(ca.iloc[-1]), float(cb.iloc[-1])
        new = next_side(self.side, z, beta, self.entry, self.exit, self.stop)
        event = PairEvent(datetime.now(timezone.utc), self.name, z, beta, "hold", pa, pb, dry_run=self.dry_run)
        if new != self.side:
            if self.side != 0.0:  # close what is open: the reverse of the entry, same amounts
                amt_a, amt_b = self.amounts
                event.action, event.amount_a, event.amount_b = "close", amt_a, amt_b
                buy_leg = 1 if self.side > 0 else 0  # long gap = long A / short B, so closing buys B back
                buy_amt, sell_amt = (amt_b, amt_a) if buy_leg == 1 else (amt_a, amt_b)
            else:
                amt_a, amt_b = self.stake / (1 + beta) / pa, self.stake * beta / (1 + beta) / pb
                event.action = "open long" if new > 0 else "open short"
                event.amount_a, event.amount_b = amt_a, amt_b
                buy_leg = 0 if new > 0 else 1
                buy_amt, sell_amt = (amt_a, amt_b) if buy_leg == 0 else (amt_b, amt_a)
            if not self.dry_run:
                event.orders = self._trade(buy_leg, buy_amt, sell_amt)
            self.side, self.amounts = new, ((amt_a, amt_b) if new != 0.0 else (0.0, 0.0))
        self.events.append(event)
        log.info("%s %s z=%+.2f β=%.2f%s", self.name, event.action, z, beta, " (dry run)" if self.dry_run else "")
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
                log.exception("pair bot step failed for %s; retrying next candle", self.name)
            n += 1
            if max_steps is not None and n >= max_steps:
                break
            time.sleep(step - time.time() % step + 2)  # wake just after the next candle closes
