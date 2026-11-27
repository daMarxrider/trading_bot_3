# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Safety limits for the live bot (#154).

All limits are optional; ``None``/``0`` switches one off. Money is in the quote currency. Drawdown is measured in
percent of ``capital`` (the bot's ``stake`` unless set), on realized plus open profit, from its peak since start.

* daily loss: halts for the rest of the UTC day (lifts itself at the next day);
* drawdown and the manual kill switch halt until :meth:`RiskGuard.resume` (or ``tradebot bot resume``);
* exposure cap, order value cap and price sanity range reject single orders;
* cool-down: pause entries for ``cooldown_minutes`` after ``cooldown_losses`` losing trades in a row;
* bad data: a profit that isn't a finite number (a NaN price) halts until the next finite one, without touching
  the peak or the day's starting point, so no limit is switched off by it.

The kill switch is a file, so the CLI, the GUI and a running bot in another process share it.
Every trigger is logged and handed to ``on_trigger`` (the hook notifications plug into).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from tradebot.config import os_config_dir

log = logging.getLogger(__name__)


def kill_file() -> Path:
    """The manual kill switch: while this file exists, live bots stop trading."""
    return os_config_dir() / "KILL"


def set_kill(on: bool = True, reason: str = "manual") -> None:
    path = kill_file()
    if on:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{reason}\n")
    else:
        path.unlink(missing_ok=True)


def kill_active() -> bool:
    return kill_file().exists()


def check_stake(stake) -> float:
    """``stake`` as a positive finite amount of money; raises ValueError (NaN would slip past ``<= 0``)."""
    try:
        value = float(stake)
    except (TypeError, ValueError):
        raise ValueError(f"the stake must be a number, not {stake!r}") from None
    if not (value > 0 and math.isfinite(value)):
        raise ValueError(f"the stake must be a positive amount, not {stake!r}")
    return value


def check_limit(name: str, value, allow_none: bool = True) -> float | None:
    """A limit as a finite number of at least 0 (0 or None: off). A NaN limit would never trigger and a negative one
    would always, both silently, so both are refused."""
    if value is None and allow_none:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"risk limit {name}: {value!r} is not a number") from None
    if not (number >= 0 and math.isfinite(number)):
        raise ValueError(f"risk limit {name}: {value!r} must be 0 (off) or a positive number")
    return number


@dataclass
class RiskLimits:
    max_daily_loss: float | None = None  #: quote currency lost in one UTC day
    max_drawdown: float | None = None  #: percent of capital below the equity peak
    max_exposure: float | None = None  #: quote value of the open position after an entry
    max_order_value: float | None = None  #: quote value of one order
    price_band: float | None = None  #: percent the exchange price may differ from the last candle close
    cooldown_losses: int = 0  #: losing trades in a row that start the cool-down
    cooldown_minutes: float = 0.0
    close_on_halt: bool = False  #: also sell the open position when a halt triggers
    capital: float | None = None  #: base for drawdown percent; default the bot's stake

    def __post_init__(self):
        for name in ("max_daily_loss", "max_drawdown", "max_exposure", "max_order_value", "price_band"):
            setattr(self, name, check_limit(name, getattr(self, name)))
        self.cooldown_minutes = check_limit("cooldown_minutes", self.cooldown_minutes, allow_none=False)
        losses = check_limit("cooldown_losses", self.cooldown_losses, allow_none=False)
        if losses != int(losses):
            raise ValueError(f"risk limit cooldown_losses: {self.cooldown_losses!r} must be a whole number")
        self.cooldown_losses = int(losses)
        if self.capital is not None:
            self.capital = check_stake(self.capital) if self.capital else None


@dataclass
class RiskEvent:
    time: datetime
    kind: str  #: daily_loss, drawdown, kill_switch, bad_data, exposure, order_size, price_band or cooldown
    message: str
    halts: bool = False  #: True when the bot stopped trading, False when one order was rejected


class RiskGuard:
    def __init__(self, limits: RiskLimits, capital: float, on_trigger: Callable[[RiskEvent], None] | None = None,
                 kill_path: Callable[[], bool] = kill_active):
        self.limits, self.capital = limits, limits.capital or capital
        self.on_trigger, self._killed = on_trigger, kill_path
        self.realized = 0.0
        self.peak = 0.0
        self.losses_in_row = 0
        self.cooldown_until: datetime | None = None
        self.halted: RiskEvent | None = None
        self._day: tuple | None = None
        self._day_start_pnl = 0.0

    def _fire(self, kind: str, message: str, halts: bool, now: datetime) -> RiskEvent:
        ev = RiskEvent(now, kind, message, halts)
        log.warning("risk guard %s: %s", kind, message)
        if halts:
            self.halted = ev
        if self.on_trigger:
            try:
                self.on_trigger(ev)
            except Exception:
                log.exception("risk trigger hook failed")
        return ev

    def resume(self) -> None:
        self.halted, self.cooldown_until, self.losses_in_row = None, None, 0
        self.peak = self.realized  # drawdown is measured from here on

    def record_trade(self, pnl: float, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        if not math.isfinite(pnl):  # added, it would make every later total NaN
            self._fire("bad_data", f"trade result {pnl!r} is not a number; not counted", False, now)
            return
        self.realized += pnl
        self.losses_in_row = self.losses_in_row + 1 if pnl < 0 else 0
        lim = self.limits
        if lim.cooldown_losses and self.losses_in_row >= lim.cooldown_losses:
            self.cooldown_until = now + timedelta(minutes=lim.cooldown_minutes)
            self.losses_in_row = 0
            self._fire("cooldown", f"{lim.cooldown_losses} losing trades in a row; no entries for "
                       f"{lim.cooldown_minutes:g} min", False, now)

    def check(self, open_pnl: float = 0.0, now: datetime | None = None) -> RiskEvent | None:
        """Evaluate the halting limits; returns the event when the bot is (still) halted, newly or not."""
        now = now or datetime.now(timezone.utc)
        lim = self.limits
        total = self.realized + open_pnl
        if not math.isfinite(total):  # nothing below may see it: NaN fails every limit check and poisons the state
            if self.halted is None:
                return self._fire("bad_data", f"open profit {open_pnl!r} is not a number (bad price data); "
                                  "no trading until prices are valid again", True, now)
            return self.halted
        if self.halted and self.halted.kind == "bad_data":
            self.halted = None  # valid prices again
        day = now.date()
        if self._day != day:
            self._day, self._day_start_pnl = day, total
            if self.halted and self.halted.kind == "daily_loss":
                self.halted = None  # a new day, a new allowance
        self.peak = max(self.peak, total)
        if self.halted:
            return self.halted
        if self._killed():
            return self._fire("kill_switch", "manual kill switch is on", True, now)
        if lim.max_daily_loss and self._day_start_pnl - total >= lim.max_daily_loss:
            return self._fire("daily_loss", f"lost {self._day_start_pnl - total:.2f} today "
                              f"(limit {lim.max_daily_loss:g})", True, now)
        if lim.max_drawdown and self.capital > 0 and (self.peak - total) / self.capital * 100 >= lim.max_drawdown:
            return self._fire("drawdown", f"drawdown {(self.peak - total) / self.capital * 100:.1f}% "
                              f"(limit {lim.max_drawdown:g}%)", True, now)
        return None

    def check_entry(self, amount: float, price: float, held: float, market_price: float | None = None,
                    now: datetime | None = None) -> RiskEvent | None:
        """Vet one buy. Returns the rejection or ``None`` when the order may go out."""
        now = now or datetime.now(timezone.utc)
        lim = self.limits
        if self.cooldown_until and now < self.cooldown_until:
            return RiskEvent(now, "cooldown", f"cool-down until {self.cooldown_until:%H:%M}")
        value = amount * price
        if (lim.max_order_value or lim.max_exposure) and not math.isfinite(held * price + value):
            # fail closed: NaN compares False against every cap and would slip through
            return self._fire("order_size", f"order value can't be computed (amount {amount!r}, price {price!r})",
                              False, now)
        if lim.max_order_value and value > lim.max_order_value:
            return self._fire("order_size", f"order value {value:.2f} above {lim.max_order_value:g}", False, now)
        if lim.max_exposure and held * price + value > lim.max_exposure:
            return self._fire("exposure", f"exposure {held * price + value:.2f} above {lim.max_exposure:g}",
                              False, now)
        if lim.price_band and market_price is not None and price > 0 \
                and not abs(market_price / price - 1) * 100 <= lim.price_band:  # NaN exchange price: rejected
            return self._fire("price_band", f"exchange price {market_price:.6g} is more than {lim.price_band:g}% "
                              f"from the last close {price:.6g}", False, now)
        return None
