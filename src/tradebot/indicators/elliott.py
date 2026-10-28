# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Elliott wave indicator (see :mod:`tradebot.analysis.elliott`)."""

from __future__ import annotations

import numpy as np

from tradebot.analysis import elliott as ew
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.core.series import OHLC
from tradebot.indicators.base import BaseIndicator
from tradebot.indicators.builtin import ATR


def to_numpy(o: OHLC) -> OHLC:
    if ops.is_torch(o.close):
        return OHLC(*(a.detach().cpu().numpy() for a in (o.open, o.high, o.low, o.close)))
    return o


def detect(o: OHLC, atr_period: int, swing_atr: float):
    """Pivots and ATR for NumPy OHLC arrays."""
    atr = np.asarray(ATR.compute(o, period=atr_period)["value"], dtype=float)
    return ew.zigzag(np.asarray(o.high, float), np.asarray(o.low, float), atr, swing_atr), atr


class ElliottWaves(BaseIndicator):
    """Automatic Elliott wave count.

    Outputs:

    * ``zigzag`` - line through the confirmed swing pivots (drawn)
    * ``wave`` - wave labels 0-5 / A-C at their pivots (drawn as text). This is a
      **hindsight** view of the whole history; do not use it for trading decisions.
    * ``state`` - **causal** wave state per bar from pivots confirmed so far:
      3/4/5 = wave 3/4/5 in progress, 10 = impulse complete, 11-13 = A/B/C;
      negative for bearish structures. Safe for algorithms and screeners.
    """

    name = "elliott"
    description = "Automatic Elliott wave count (zigzag, wave labels, causal wave state)"
    params = {"atr_period": IntParam(14, 5, 50), "swing_atr": Param(4.0, 1.0, 10.0), "min_score": Param(0.25, 0.0, 0.9)}
    outputs = ("zigzag", "wave", "state")
    labels = {"wave": ew.LABELS}
    label_anchor = "zigzag"
    hidden = ("state",)

    @classmethod
    def compute(cls, o, atr_period, swing_atr, min_score):
        n = len(o)
        pivots, _ = detect(to_numpy(o), atr_period, swing_atr)
        zz = np.full(n, np.nan)
        if len(pivots) >= 2:
            idx = np.array([p.index for p in pivots])
            zz[idx[0]: idx[-1] + 1] = np.interp(np.arange(idx[0], idx[-1] + 1), idx, [p.price for p in pivots])
        elif pivots:
            zz[pivots[0].index] = pivots[0].price
        wave = np.full(n, np.nan)
        for pivot, code in ew.label_waves(pivots, min_score):
            wave[pivot.index] = code
        state = ew.wave_states(pivots, n).state.astype(float)
        return {k: ops.asarray(v, o.close) for k, v in (("zigzag", zz), ("wave", wave), ("state", state))}
