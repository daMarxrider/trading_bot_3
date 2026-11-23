# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Worked example for docs/indicators-as-algorithms.md: an indicator and an algorithm built on it.

Copy to ~/.config/tradebot/plugins/ (or point $TRADEBOT_PLUGIN_DIR here). It registers

* the indicator ``zscore``: how many standard deviations the close is from its moving average;
* the algorithm ``zscore_reversion``: buy when the z-score drops below ``-entry``, sell when it is back above
  ``-exit``. It calls ``ZScore.compute`` and uses ``ops`` only, so it is vectorized: the GPU/batched
  optimizer, ``vote``, ``pool`` and the combinator can all use it.

    tradebot list algorithms
    tradebot backtest BTC/USDT zscore_reversion -p entry=2.2
    tradebot optimize BTC/USDT zscore_reversion
"""

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.base import BaseIndicator, Style


class ZScore(BaseIndicator):
    name = "zscore"
    description = "Distance of the close from its moving average, in standard deviations"
    params = {"period": IntParam(20, 5, 200, "Candles in the average and the standard deviation.")}
    outputs = ("value", "oversold")
    overlay = False                      # its own pane under the price
    levels = (-2.0, 0.0, 2.0)            # reference lines
    styles = {"value": Style(color="accent", width=2)}
    alerts = {"oversold": "The z-score is below -2"}   # 0/1 per bar; hidden from the chart, usable in the Scanner

    @classmethod
    def compute(cls, o, period):
        mean = ops.sma(o.close, period)
        std = ops.rolling_std(o.close, period)
        z = (o.close - mean) / std       # NaN while warming up
        return {"value": z, "oversold": ops.flag(z < -2.0, z)}


class ZScoreReversion(BaseAlgorithm):
    name = "zscore_reversion"
    description = "Buy when the close is `entry` standard deviations below its average, sell when it is back near it"
    params = {
        "period": IntParam(20, 5, 200, "Candles in the average the z-score is measured against."),
        "entry": Param(2.0, 0.5, 4.0, "Buy when the z-score falls below minus this."),
        "exit": Param(0.0, -1.0, 1.0, "Sell when the z-score is back above this."),
    }

    def signals(self, o, p):
        z = ZScore.compute(o, period=p["period"])["value"]   # the indicator does the maths
        # z[t] only uses candles up to t, so deciding at the close of t is not look-ahead (see the guide)
        return ops.latch(z < -p["entry"], z > p["exit"])

    def chart_indicators(self):
        return [ZScore(period=self.values["period"])]
