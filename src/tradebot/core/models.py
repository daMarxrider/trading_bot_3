# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Plain data types shared by brokers, data providers and the bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd

OHLC_COLUMNS = ["open", "high", "low", "close"]


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    #: Semi-automated brokers: the order was handed to the user for manual execution.
    PENDING_MANUAL = "pending_manual"


class ExecutionMode(str, Enum):
    LIVE = "live"  #: orders are executed through an API
    SEMI_AUTO = "semi_auto"  #: orders become intents the user confirms/executes manually
    READ_ONLY = "read_only"  #: tracking only, no order placement


@dataclass
class Ticker:
    symbol: str
    last: float
    bid: float | None = None
    ask: float | None = None
    timestamp: datetime | None = None


@dataclass
class Order:
    id: str
    symbol: str
    side: Side
    type: OrderType
    amount: float
    price: float | None = None
    status: OrderStatus = OrderStatus.OPEN
    filled: float = 0.0
    timestamp: datetime | None = None
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class Balance:
    asset: str
    free: float
    used: float = 0.0

    @property
    def total(self) -> float:
        return self.free + self.used


@dataclass
class Position:
    """A holding as tracked by a portfolio (broker or manual import)."""

    symbol: str
    quantity: float
    avg_cost: float | None = None  #: per-unit cost basis, enables Yield on Cost
    asset_class: AssetClass = AssetClass.STOCK


def guess_asset_class(symbol: str) -> AssetClass:
    """``BTC/USDT``-style pairs are crypto, everything else is treated as a stock/ETF ticker."""
    return AssetClass.CRYPTO if "/" in symbol else AssetClass.STOCK


def to_ohlc_frame(rows) -> pd.DataFrame:
    """Build a normalized OHLC frame (UTC DatetimeIndex, float columns) from raw rows.

    ``rows`` is a sequence of ``[timestamp_ms, open, high, low, close, (volume)]``.
    Volume is kept when present for screening only; charting, indicators and
    backtests read just the OHLC columns.
    """
    rows = list(rows)
    columns = ["timestamp", *OHLC_COLUMNS, "volume"]
    df = pd.DataFrame(rows, columns=columns[: len(rows[0])] if rows else columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return normalize_ohlc(df.set_index("timestamp"))


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, keep OHLC (+volume), UTC index, sorted, deduplicated, no NaN rows."""
    df = df.rename(columns=str.lower)
    keep = OHLC_COLUMNS + (["volume"] if "volume" in df.columns else [])
    missing = set(OHLC_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"OHLC frame lacks columns: {sorted(missing)}")
    df = df[keep].astype(float)
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=OHLC_COLUMNS)
