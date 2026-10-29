# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Crypto exchanges, all driven through ccxt."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar

from tradebot.brokers.base import BaseBroker
from tradebot.core.models import AssetClass, Balance, Order, OrderStatus, OrderType, Side, Ticker, to_ohlc_frame

_STATUS = {"open": OrderStatus.OPEN, "closed": OrderStatus.CLOSED, "canceled": OrderStatus.CANCELED}


class CcxtBroker(BaseBroker):
    """Live-trading adapter for any ccxt exchange. Subclasses only set ``name``/``exchange_id``."""

    exchange_id: ClassVar[str]
    asset_classes = frozenset({AssetClass.CRYPTO})
    credential_keys = ("api_key", "secret")

    def __init__(self, credentials=None, data=None, sandbox: bool = False, exchange=None):
        super().__init__(credentials, data)
        if exchange is None:
            import ccxt

            if not hasattr(ccxt, self.exchange_id):
                raise NotImplementedError(f"ccxt {ccxt.__version__} has no exchange {self.exchange_id!r}")
            config = {"enableRateLimit": True}
            mapping = {"api_key": "apiKey", "secret": "secret", "password": "password", "uid": "uid"}
            config.update({mapping[k]: v for k, v in self.credentials.items() if k in mapping and v})
            exchange = getattr(ccxt, self.exchange_id)(config)
            if sandbox:
                exchange.set_sandbox_mode(True)
        self.exchange = exchange

    def get_ticker(self, symbol: str) -> Ticker:
        t = self.exchange.fetch_ticker(symbol)
        ts = datetime.fromtimestamp(t["timestamp"] / 1000, timezone.utc) if t.get("timestamp") else None
        return Ticker(symbol, float(t["last"]), t.get("bid"), t.get("ask"), ts)

    def get_klines(self, symbol: str, timeframe: str = "1d", limit: int = 500):
        rows = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return to_ohlc_frame(rows) if rows else super().get_klines(symbol, timeframe, limit)

    def place_order(self, symbol, side, amount, type=OrderType.MARKET, price=None) -> Order:
        type, side = OrderType(type), Side(side)
        if type is OrderType.LIMIT and price is None:
            raise ValueError("limit orders need a price")
        self.exchange.load_markets()
        amount = float(self.exchange.amount_to_precision(symbol, amount))
        if price is not None:
            price = float(self.exchange.price_to_precision(symbol, price))
        raw = self.exchange.create_order(symbol, type.value, side.value, amount, price)
        return self._order(raw, symbol, side, type, amount, price)

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        self.exchange.cancel_order(order_id, symbol)
        return True

    def get_balance(self) -> dict[str, Balance]:
        bal = self.exchange.fetch_balance()
        free, used = bal.get("free", {}), bal.get("used", {})
        return {
            asset: Balance(asset, float(free.get(asset) or 0), float(used.get(asset) or 0))
            for asset in set(free) | set(used)
            if (free.get(asset) or 0) or (used.get(asset) or 0)
        }

    @staticmethod
    def _order(raw, symbol, side, type, amount, price) -> Order:
        ts = raw.get("timestamp")
        return Order(
            id=str(raw["id"]),
            symbol=raw.get("symbol") or symbol,
            side=side,
            type=type,
            amount=float(raw.get("amount") or amount),
            price=raw.get("average") or raw.get("price") or price,
            status=_STATUS.get(raw.get("status"), OrderStatus.OPEN),
            filled=float(raw.get("filled") or 0),
            timestamp=datetime.fromtimestamp(ts / 1000, timezone.utc) if ts else None,
            raw=raw,
        )


class BinanceBroker(CcxtBroker):
    name = "binance"
    exchange_id = "binance"


class KucoinBroker(CcxtBroker):
    name = "kucoin"
    title = "KuCoin"
    exchange_id = "kucoin"
    credential_keys = ("api_key", "secret", "password")  # KuCoin API passphrase


class PionexBroker(CcxtBroker):
    """Extension point: activates automatically once the installed ccxt ships a ``pionex`` exchange."""

    name = "pionex"
    exchange_id = "pionex"
