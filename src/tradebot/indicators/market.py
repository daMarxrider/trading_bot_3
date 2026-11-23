# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Crypto market data as indicators (#166): funding rate, open interest, Fear & Greed, MVRV and active addresses.

They read the market data columns of the candles (:mod:`tradebot.data.market` fetches and lines them up with the
candles without look-ahead: a bar sees only what was known at its open). Without that data, say for a stock or
offline without a cache, every output is NaN, so rules built on them never fire. They work everywhere an indicator
does (pane under the chart, combinator rules such as ``funding > 0.05 & rsi < 30``, ``evolve --market-data``), but
not on a higher timeframe (``tf=``) and not in the Scanner.
"""

from __future__ import annotations

from tradebot.core import ops
from tradebot.core.registry import IntParam
from tradebot.indicators.base import BaseIndicator, Style


def _smooth(x, n: int):
    return x if n <= 1 else ops.sma(x, n)


class Funding(BaseIndicator):
    name = "funding"
    description = "Perpetual futures funding rate in percent per settlement (positive: longs pay shorts)"
    params = {"smooth": IntParam(1, 1, 100, "candles averaged (1 = the raw rate)")}
    overlay = False
    levels = (-0.05, 0.0, 0.05)
    market_data = ("funding_rate",)
    styles = {"value": Style(color="accent", width=1.6)}

    @classmethod
    def compute(cls, o, smooth):
        return {"value": _smooth(o.data("funding_rate") * 100.0, smooth)}


class OpenInterest(BaseIndicator):
    name = "open_interest"
    description = "Open interest of the perpetual futures, in the quote currency"
    params = {"smooth": IntParam(1, 1, 100, "candles averaged (1 = the raw value)")}
    overlay = False
    market_data = ("open_interest",)
    units = {"value": "open_interest"}
    styles = {"value": Style(color="series[1]", width=1.6)}

    @classmethod
    def compute(cls, o, smooth):
        return {"value": _smooth(o.data("open_interest"), smooth)}


class OpenInterestChange(BaseIndicator):
    name = "oi_change"
    description = "Change of the open interest over a number of candles, in percent"
    params = {"period": IntParam(24, 1, 500, "candles the change is measured over")}
    overlay = False
    levels = (-10.0, 0.0, 10.0)
    market_data = ("open_interest",)
    styles = {"value": Style(color="series[1]", width=1.6)}

    @classmethod
    def compute(cls, o, period):
        oi = o.data("open_interest")
        return {"value": (oi / ops.shift(oi, period) - 1.0) * 100.0}


class FearGreed(BaseIndicator):
    name = "fear_greed"
    description = "Crypto Fear & Greed index (alternative.me), 0 = extreme fear, 100 = extreme greed"
    params = {"smooth": IntParam(1, 1, 60, "candles averaged (1 = the daily value)")}
    overlay = False
    levels = (25.0, 75.0)
    level_styles = {25.0: Style(color="down", dash="dash"), 75.0: Style(color="up", dash="dash")}
    market_data = ("fear_greed",)
    styles = {"value": Style(color="accent", width=1.6)}

    @classmethod
    def compute(cls, o, smooth):
        return {"value": _smooth(o.data("fear_greed"), smooth)}


class MVRV(BaseIndicator):
    name = "mvrv"
    description = "Market value to realized value of the coin (Coin Metrics): above 3.5 is historically hot, below 1 cheap"
    params = {"smooth": IntParam(1, 1, 60, "candles averaged (1 = the daily value)")}
    overlay = False
    levels = (1.0, 3.5)
    level_styles = {1.0: Style(color="down", dash="dash"), 3.5: Style(color="up", dash="dash")}
    market_data = ("mvrv",)
    styles = {"value": Style(color="accent", width=1.6)}

    @classmethod
    def compute(cls, o, smooth):
        return {"value": _smooth(o.data("mvrv"), smooth)}


class ActiveAddresses(BaseIndicator):
    name = "active_addresses"
    description = "Addresses active on the coin's chain per day (Coin Metrics)"
    params = {"smooth": IntParam(1, 1, 60, "candles averaged (1 = the daily value)")}
    overlay = False
    market_data = ("active_addresses",)
    units = {"value": "addresses"}
    styles = {"value": Style(color="series[2]", width=1.6)}

    @classmethod
    def compute(cls, o, smooth):
        return {"value": _smooth(o.data("active_addresses"), smooth)}
