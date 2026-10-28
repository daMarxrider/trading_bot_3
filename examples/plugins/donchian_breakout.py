# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Example plugin: copy to ~/.config/tradebot/plugins/ (or point $TRADEBOT_PLUGIN_DIR here).

It registers one indicator and one algorithm; both immediately appear in
`tradebot list`, the GUI chart/backtest lab, and can be GPU-optimized.
"""

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.base import BaseIndicator, Style
from tradebot.indicators.builtin import ATR, Donchian


class RateOfChange(BaseIndicator):
    name = "roc"
    description = "Rate of change of the close, in percent"
    params = {"period": IntParam(10, 1, 200)}
    overlay = False
    levels = (0.0,)
    styles = {"value": Style(color="accent", width=2)}  # optional: colour, width and dash per output

    @classmethod
    def compute(cls, o, period):
        prev = ops.shift(o.close, period)
        return {"value": 100 * (o.close / prev - 1)}


class DonchianBreakout(BaseAlgorithm):
    name = "donchian_breakout"
    description = "Buy a breakout above the N-bar high, exit below a trailing ATR stop"
    params = {"period": IntParam(55, 10, 200), "atr_mult": Param(3.0, 1.0, 6.0)}

    def signals(self, o, p):
        ch = Donchian.compute(o, period=p["period"])
        atr = ATR.compute(o, period=14)["value"]
        enter = o.close > ops.shift(ch["upper"], 1)
        exit_ = o.close < ops.shift(ch["upper"], 1) - p["atr_mult"] * atr
        return ops.latch(enter, exit_)

    def chart_indicators(self):
        return [Donchian(period=self.values["period"])]
