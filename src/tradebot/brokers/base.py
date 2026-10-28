# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Broker interface.

Every broker exposes the same five calls - ``get_ticker``, ``get_klines``,
``place_order``, ``cancel_order``, ``get_balance`` - plus an
:class:`~tradebot.core.models.ExecutionMode`:

* ``LIVE``: orders go to the venue's API (Binance, KuCoin, any ccxt exchange).
* ``SEMI_AUTO``: the venue has no trading API; ``place_order`` records an
  :class:`OrderIntent` for the user to execute by hand (Scalable Capital,
  Trade Republic).
* ``READ_ONLY``: tracking only; ``place_order`` raises.

Adding a broker::

    class Bitstamp(CcxtBroker):          # any ccxt exchange: one attribute
        name = "bitstamp"
        exchange_id = "bitstamp"

or subclass :class:`BaseBroker` / :class:`ManualExecutionBroker` for venues
without ccxt support. Defining the class registers it under ``name``.
"""

from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, ClassVar

import pandas as pd

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
)
from tradebot.core.registry import Registry
from tradebot.core.timeframes import to_timedelta

log = logging.getLogger(__name__)
BROKERS: Registry["BaseBroker"] = Registry("broker")


class ReadOnlyBrokerError(RuntimeError):
    pass


class BaseBroker(ABC):
    name: ClassVar[str]
    execution_mode: ClassVar[ExecutionMode] = ExecutionMode.LIVE
    asset_classes: ClassVar[frozenset[AssetClass]] = frozenset({AssetClass.CRYPTO})
    #: Credential keys this broker understands (documentation + config validation).
    credential_keys: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BROKERS.register(cls)

    def __init__(self, credentials: dict | None = None, data=None):
        self.credentials = credentials or {}
        self._data = data

    @property
    def data(self):
        """Shared :class:`~tradebot.data.MarketDataEngine` (created lazily)."""
        if self._data is None:
            from tradebot.data import MarketDataEngine

            self._data = MarketDataEngine()
        return self._data

    @property
    def can_trade(self) -> bool:
        return self.execution_mode is not ExecutionMode.READ_ONLY

    @abstractmethod
    def get_ticker(self, symbol: str) -> Ticker: ...

    def get_klines(self, symbol: str, timeframe: str = "1d", limit: int = 500) -> pd.DataFrame:
        """OHLC candles; the default implementation uses the market data engine
        (prices are assumed identical across venues)."""
        end = pd.Timestamp.now(tz="UTC")
        return self.data.get_ohlc(symbol, timeframe, start=end - limit * to_timedelta(timeframe), end=end).tail(limit)

    @abstractmethod
    def place_order(
        self, symbol: str, side: Side, amount: float, type: OrderType = OrderType.MARKET, price: float | None = None
    ) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool: ...

    @abstractmethod
    def get_balance(self) -> dict[str, Balance]: ...

    def get_positions(self) -> list[Position]:
        """Non-zero holdings. Brokers with cost-basis data override this."""
        return [
            Position(asset, bal.total, asset_class=AssetClass.CRYPTO)
            for asset, bal in self.get_balance().items()
            if bal.total > 0
        ]


@dataclass
class OrderIntent:
    """An order a SEMI_AUTO broker asks the user to execute manually."""

    id: str
    broker: str
    symbol: str
    side: Side
    amount: float
    type: OrderType
    price: float | None
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    #: Broker-specific data, e.g. a pre-trade preview and its confirmation id.
    details: dict = field(default_factory=dict, repr=False)

    def describe(self) -> str:
        at = f" @ {self.price}" if self.price else " at market"
        return f"[{self.broker}] {self.side.value.upper()} {self.amount:g} {self.symbol}{at}"


class ManualExecutionBroker(BaseBroker):
    """Base for venues without a trading API.

    Orders become :class:`OrderIntent` objects kept in :attr:`pending` and
    passed to ``on_intent`` (e.g. a desktop notification or a GUI dialog).
    """

    execution_mode = ExecutionMode.SEMI_AUTO
    asset_classes = frozenset({AssetClass.STOCK})
    _ids = itertools.count(1)

    def __init__(self, credentials=None, data=None, on_intent: Callable[[OrderIntent], None] | None = None):
        super().__init__(credentials, data)
        self.pending: dict[str, OrderIntent] = {}
        self.on_intent = on_intent

    def get_ticker(self, symbol: str) -> Ticker:
        return Ticker(symbol, self.data.latest_price(symbol))

    def place_order(self, symbol, side, amount, type=OrderType.MARKET, price=None) -> Order:
        if self.execution_mode is ExecutionMode.READ_ONLY:
            raise ReadOnlyBrokerError(f"{self.name} is read-only")
        intent = OrderIntent(f"{self.name}-{next(self._ids)}", self.name, symbol, Side(side), amount, OrderType(type), price)
        self.pending[intent.id] = intent
        log.warning("manual order required: %s", intent.describe())
        if self.on_intent:
            self.on_intent(intent)
        return Order(intent.id, symbol, intent.side, intent.type, amount, price, OrderStatus.PENDING_MANUAL,
                     timestamp=intent.created)

    def cancel_order(self, order_id, symbol=None) -> bool:
        return self.pending.pop(order_id, None) is not None

    def confirm(self, intent_id: str, **kwargs) -> Order:
        """Submit a pending intent after the user explicitly approved it.

        Must only be called from a direct user action (CLI prompt, GUI button),
        never automatically. Brokers without any submission API cannot do this.
        """
        raise NotImplementedError(f"{self.name} has no order API; execute {intent_id} manually")
