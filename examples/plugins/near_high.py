# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Example scanner script (docs/scanner.md): distance to the N-bar high, with two alerts."""

from tradebot.core import ops
from tradebot.core.registry import IntParam
from tradebot.indicators.base import BaseIndicator


class NearHigh(BaseIndicator):
    name = "near_high"
    description = "How close the close is to its N-bar high, with a breakout alert"
    params = {"period": IntParam(20, 2, 200)}
    outputs = ("high_n", "pct_below", "breakout", "near")
    alerts = {"breakout": "Close makes a new N-bar high", "near": "Close within 2% of the N-bar high"}
    units = {"pct_below": "ratio"}

    @classmethod
    def compute(cls, o, period):
        high_n = ops.rolling_max(o.high, period)
        prev = ops.shift(high_n, 1)  # the high before this bar, so no look-ahead
        pct_below = (high_n - o.close) / high_n
        return {
            "high_n": high_n,
            "pct_below": pct_below,
            "breakout": ops.flag(o.close > prev, prev),
            "near": ops.flag(pct_below <= 0.02, pct_below),
        }
