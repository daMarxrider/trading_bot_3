# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Paper trading: a simulated account that survives restarts.

Orders fill at the last price of the real data source, minus the same proportional ``fee`` as the backtester
(a buy pays ``price * (1 + fee)`` per unit, a sell receives ``price * (1 - fee)``). Cash, positions and the
order history live in a JSON file next to the app's other state, written atomically after every order, so
``tradebot bot run --paper`` resumes where it stopped.

Config (all optional)::

    [brokers.paper]
    starting_balance = 10000
    currency = "USDT"     # cash currency; also the quote of symbols without one, like AAPL
    fee = 0.001
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from tradebot.brokers.base import BaseBroker
from tradebot.core.models import (
    AssetClass,
    Balance,
    ExecutionMode,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Ticker,
    guess_asset_class,
)

log = logging.getLogger(__name__)
DEFAULT_BALANCE, DEFAULT_FEE, DEFAULT_CURRENCY = 10_000.0, 0.001, "USDT"


def default_account_path() -> Path:
    from tradebot.config import os_config_dir

    return os_config_dir() / "paper_account.json"


class PaperBroker(BaseBroker):
    name = "paper"
    title = "Paper trading"
    execution_mode = ExecutionMode.LIVE  # orders are accepted (and simulated)
    asset_classes = frozenset({AssetClass.CRYPTO, AssetClass.STOCK})
    credential_keys = ("starting_balance", "currency", "fee")
    simulated = True

    def __init__(self, credentials=None, data=None, path: Path | str | None = None, starting_balance: float | None = None,
                 fee: float | None = None, currency: str | None = None):
        super().__init__(credentials, data)
        c = self.credentials
        self.path = Path(path) if path else default_account_path()
        self.fee = float(fee if fee is not None else c.get("fee") or DEFAULT_FEE)
        self.currency = str(currency or c.get("currency") or DEFAULT_CURRENCY)
        self.starting_balance = float(starting_balance if starting_balance is not None
                                      else c.get("starting_balance") or DEFAULT_BALANCE)
        self.cash = self.starting_balance
        self.positions: dict[str, dict] = {}  #: base asset -> {"quantity", "avg_cost"}
        self.orders: list[dict] = []
        self.created = datetime.now(timezone.utc).isoformat()
        self._load()

    # --- persistence -------------------------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            s = json.loads(self.path.read_text())
            self.starting_balance, self.cash = float(s["starting_balance"]), float(s["cash"])
            self.currency, self.fee = str(s["currency"]), float(s["fee"])
            self.positions, self.orders, self.created = s["positions"], s["orders"], s["created"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"paper account file {self.path} is unreadable ({exc}); fix or delete it "
                               "(`tradebot paper reset`)") from exc

    def save(self) -> None:
        """Write the account atomically (temp file in the same folder, then rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = {"starting_balance": self.starting_balance, "cash": self.cash, "currency": self.currency,
                 "fee": self.fee, "created": self.created, "positions": self.positions, "orders": self.orders}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(state, fh, indent=1)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def reset(self, starting_balance: float | None = None) -> None:
        """Start over with a fresh account (the file is rewritten)."""
        if starting_balance is not None:
            self.starting_balance = float(starting_balance)
        self.cash, self.positions, self.orders = self.starting_balance, {}, []
        self.created = datetime.now(timezone.utc).isoformat()
        self.save()

    # --- simulation --------------------------------------------------------------------------------------
    @staticmethod
    def _base(symbol: str) -> str:
        return symbol.split("/")[0]

    def get_ticker(self, symbol: str) -> Ticker:
        return Ticker(symbol, self.data.latest_price(symbol))

    def fill_price(self, symbol: str, side: Side, price: float) -> float:
        """Price an order really fills at. Hook for a slippage model (#146): move ``price`` against the trader."""
        return price

    def place_order(self, symbol, side, amount, type=OrderType.MARKET, price=None) -> Order:
        side, type = Side(side), OrderType(type)
        if type is not OrderType.MARKET:
            raise NotImplementedError("the paper broker only fills market orders")
        if amount <= 0:
            raise ValueError("amount must be positive")
        base = self._base(symbol)
        px = self.fill_price(symbol, side, float(self.get_ticker(symbol).last))
        pos = self.positions.get(base, {"quantity": 0.0, "avg_cost": 0.0})
        fee = amount * px * self.fee
        profit = None
        if side is Side.BUY:
            cost = amount * px + fee
            if cost > self.cash + 1e-9:
                raise ValueError(f"not enough cash: need {cost:.2f} {self.currency}, have {self.cash:.2f}")
            qty = pos["quantity"] + amount
            pos = {"quantity": qty, "avg_cost": (pos["quantity"] * pos["avg_cost"] + amount * px) / qty}
            self.cash -= cost
        else:
            if amount > pos["quantity"] + 1e-12:
                raise ValueError(f"not enough {base}: want {amount:g}, have {pos['quantity']:g}")
            profit = amount * (px - pos["avg_cost"]) - fee
            self.cash += amount * px - fee
            pos = {"quantity": pos["quantity"] - amount, "avg_cost": pos["avg_cost"]}
        if pos["quantity"] > 1e-12:
            self.positions[base] = pos
        else:
            self.positions.pop(base, None)
        now = datetime.now(timezone.utc)
        oid = f"paper-{len(self.orders) + 1}"
        self.orders.append({"id": oid, "time": now.isoformat(), "symbol": symbol, "side": side.value, "amount": amount,
                            "price": px, "fee": fee, "profit": profit})
        self.save()
        log.info("paper %s %g %s @ %.6g (fee %.4g)", side.value, amount, symbol, px, fee)
        return Order(oid, symbol, side, type, amount, px, OrderStatus.CLOSED, amount, now)

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        return False  # market orders fill at once; nothing is ever open

    def get_balance(self) -> dict[str, Balance]:
        out = {self.currency: Balance(self.currency, self.cash)}
        out.update({a: Balance(a, p["quantity"]) for a, p in self.positions.items()})
        return out

    def _symbol_of(self, asset: str) -> str:
        return next((o["symbol"] for o in reversed(self.orders) if self._base(o["symbol"]) == asset), asset)

    def get_positions(self) -> list[Position]:
        return [Position(a, p["quantity"], p["avg_cost"], guess_asset_class(self._symbol_of(a)))
                for a, p in self.positions.items()]

    def summary(self) -> dict:
        """Equity and P&L at current prices (positions whose price can't be fetched count at cost)."""
        value = 0.0
        for a, p in self.positions.items():
            try:
                px = float(self.get_ticker(self._symbol_of(a)).last)
            except Exception as exc:
                log.warning("no price for %s (%s); valued at cost", a, exc)
                px = p["avg_cost"]
            value += p["quantity"] * px
        equity = self.cash + value
        return {"cash": self.cash, "positions_value": value, "equity": equity, "starting_balance": self.starting_balance,
                "pnl": equity - self.starting_balance, "pnl_pct": equity / self.starting_balance - 1,
                "realised_pnl": sum(o["profit"] or 0.0 for o in self.orders), "fees": sum(o["fee"] for o in self.orders),
                "orders": len(self.orders), "currency": self.currency}
