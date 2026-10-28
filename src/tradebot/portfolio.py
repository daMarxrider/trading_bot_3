# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Portfolio valuation, allocation and dividend metrics.

* **Current dividend yield** = forward annual dividend / price
* **Annual (trailing) dividend yield** = dividends paid in the last 12 months / price
* **Yield on cost (YoC)** = forward annual dividend / average purchase price
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from tradebot.core.models import AssetClass, Position

log = logging.getLogger(__name__)
STABLECOINS = {"USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USD"}


@dataclass
class DividendInfo:
    forward_annual: float = 0.0  #: expected dividend per share over the next year
    trailing_annual: float = 0.0  #: paid per share during the last 12 months


@dataclass
class Holding:
    symbol: str
    quantity: float
    price: float
    avg_cost: float | None
    dividends: DividendInfo
    asset_class: AssetClass

    @property
    def value(self) -> float:
        return self.quantity * self.price

    @property
    def cost(self) -> float | None:
        return self.quantity * self.avg_cost if self.avg_cost else None

    @property
    def current_yield(self) -> float:
        return self.dividends.forward_annual / self.price if self.price else 0.0

    @property
    def annual_yield(self) -> float:
        return self.dividends.trailing_annual / self.price if self.price else 0.0

    @property
    def yield_on_cost(self) -> float | None:
        return self.dividends.forward_annual / self.avg_cost if self.avg_cost else None

    @property
    def annual_income(self) -> float:
        return self.quantity * self.dividends.forward_annual


@dataclass
class PortfolioSummary:
    holdings: list[Holding]

    @property
    def total_value(self) -> float:
        return sum(h.value for h in self.holdings)

    @property
    def annual_income(self) -> float:
        return sum(h.annual_income for h in self.holdings)

    @property
    def current_yield(self) -> float:
        return self.annual_income / self.total_value if self.total_value else 0.0

    @property
    def annual_yield(self) -> float:
        paid = sum(h.quantity * h.dividends.trailing_annual for h in self.holdings)
        return paid / self.total_value if self.total_value else 0.0

    @property
    def yield_on_cost(self) -> float | None:
        with_cost = [h for h in self.holdings if h.cost]
        cost = sum(h.cost for h in with_cost)
        return sum(h.annual_income for h in with_cost) / cost if cost else None

    def allocation(self, by: str = "symbol") -> pd.Series:
        """Weights by ``symbol`` or ``asset_class``."""
        s = pd.Series({h.symbol: h.value for h in self.holdings}, dtype=float)
        if by == "asset_class":
            s = pd.Series([h.value for h in self.holdings], index=[h.asset_class.value for h in self.holdings]).groupby(level=0).sum()
        total = s.sum()
        return (s / total).sort_values(ascending=False) if total else s

    def to_frame(self) -> pd.DataFrame:
        alloc = self.allocation()
        return pd.DataFrame(
            [
                {
                    "symbol": h.symbol, "quantity": h.quantity, "price": h.price, "value": h.value,
                    "weight": alloc.get(h.symbol, 0.0), "avg_cost": h.avg_cost,
                    "current_yield": h.current_yield, "annual_yield": h.annual_yield,
                    "yield_on_cost": h.yield_on_cost, "annual_income": h.annual_income,
                }
                for h in self.holdings
            ]
        )


def yf_dividends(symbol: str) -> DividendInfo:
    """Dividend data from Yahoo Finance (zeros for non-payers and crypto)."""
    if "/" in symbol:
        return DividendInfo()
    import yfinance as yf

    t = yf.Ticker(symbol)
    divs = t.dividends
    trailing = 0.0
    if divs is not None and len(divs):
        cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.DateOffset(years=1)
        trailing = float(divs[divs.index > cutoff].sum())
    try:
        forward = float(t.info.get("dividendRate") or 0.0)
    except Exception:  # .info is flaky; fall back to trailing
        forward = 0.0
    return DividendInfo(forward or trailing, trailing)


def price_symbol(position: Position) -> str:
    """Exchange balances report bare assets (``BTC``); price them against USDT."""
    if position.asset_class is AssetClass.CRYPTO and "/" not in position.symbol:
        return f"{position.symbol}/USDT"
    return position.symbol


def build_portfolio(
    positions: list[Position],
    price: Callable[[str], float],
    dividends: Callable[[str], DividendInfo] = yf_dividends,
) -> PortfolioSummary:
    holdings = []
    for pos in positions:
        sym = price_symbol(pos)
        try:
            px = 1.0 if pos.symbol in STABLECOINS else price(sym)
        except Exception as exc:
            log.warning("no price for %s: %s", sym, exc)
            continue
        try:
            div = dividends(pos.symbol) if pos.asset_class is AssetClass.STOCK else DividendInfo()
        except Exception as exc:
            log.warning("no dividend data for %s: %s", pos.symbol, exc)
            div = DividendInfo()
        holdings.append(Holding(pos.symbol, pos.quantity, px, pos.avg_cost, div, pos.asset_class))
    return PortfolioSummary(holdings)
