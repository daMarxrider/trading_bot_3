# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Market regime detection (#168): trend (bull / bear / sideways) and volatility (high / low) per bar.

The trend regime compares the close with the close ``period`` candles ago: more than ``threshold`` percent up is
bull, more than that down is bear, anything between is sideways. The volatility regime is high while the ATR (as a
share of the close) is above its own average over five times as many candles, low otherwise. Only current and
earlier candles are used, so the regime on a bar never changes when later candles arrive.

It is an ordinary indicator, so it works everywhere one does: in the combinator as a filter
(``rsi_threshold & regime.bull``, ``regime.low_vol``, ``regime > 0``), as a primitive in ``evolve``, on the chart
as a background shade, and for per-regime metrics (:func:`tradebot.backtest.report.regime_table`).
"""

from __future__ import annotations

from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.base import BaseIndicator
from tradebot.indicators.builtin import ATR

#: Background colours of the chart shade, ``{output: {code: colour}}``; regimes without a colour stay unshaded.
SHADES = {"trend": {1.0: "rgba(38, 166, 154, 0.13)", -1.0: "rgba(239, 83, 80, 0.13)"},
          "volatility": {1.0: "rgba(255, 183, 77, 0.10)"}}
#: Regime names as shown in reports, in this order.
NAMES = {"bull": "bull", "bear": "bear", "sideways": "sideways", "high_vol": "high volatility",
         "low_vol": "low volatility"}


class Regime(BaseIndicator):
    name = "regime"
    category = "trend"
    description = "Market regime: bull / bear / sideways trend and high / low volatility"
    params = {"period": IntParam(50, 5, 500, "candles the trend is measured over"),
              "threshold": Param(5.0, 0.5, 50.0, "percent the close must have risen (fallen) over the period to "
                                                 "count as bull (bear); in between is sideways"),
              "vol_period": IntParam(14, 2, 100, "candles of the ATR; volatility is high above its average over "
                                                 "five times as many candles")}
    outputs = ("trend", "volatility", "bull", "bear", "sideways", "high_vol", "low_vol")
    overlay = True
    hidden = ("trend", "volatility")
    levels = (-1.0, 1.0)
    units = {"trend": "regime", "volatility": "regime"}
    shades = SHADES
    alerts = {"bull": "bull regime (the close is up more than the threshold over the period)",
              "bear": "bear regime (the close is down more than the threshold over the period)",
              "sideways": "sideways regime (neither bull nor bear)",
              "high_vol": "high volatility (ATR above its longer average)",
              "low_vol": "low volatility (ATR at or below its longer average)"}

    @classmethod
    def compute(cls, o, period, threshold, vol_period):
        ret = o.close / ops.shift(o.close, period) - 1.0
        lim = threshold / 100.0
        trend = ops.where(ops.isnan(ret), ops.NAN, ops.where(ret > lim, 1.0, ops.where(ret < -lim, -1.0, 0.0)))
        rel = ATR.compute(o, period=vol_period)["value"] / o.close
        base = ops.sma(rel, vol_period * 5)
        vol = ops.where(ops.isnan(base) | ops.isnan(rel), ops.NAN, ops.where(rel > base, 1.0, 0.0))
        return {"trend": trend, "volatility": vol,
                "bull": ops.flag(trend > 0, trend), "bear": ops.flag(trend < 0, trend),
                "sideways": ops.flag(trend == 0, trend),
                "high_vol": ops.flag(vol > 0, vol), "low_vol": ops.flag(vol == 0, vol)}
