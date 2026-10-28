# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Elliott wave strategy: trade wave 3 (and optionally wave 5) of bullish impulses.

Setups, decided at the close of each bar from pivots confirmed so far:

* **wave 3**: waves 1-2 complete (wave 2 retraced between ``wave2_min`` and
  ``wave2_max`` of wave 1, Fibonacci score >= ``min_score``). Enter when the
  close breaks above the wave-1 high. Stop: wave-2 low minus ``stop_atr`` x ATR.
  Target: wave-2 low + ``target_ext`` x wave 1.
* **wave 5** (``trade_wave5 = 1``): waves 1-4 complete under the impulse rules.
  Enter on a close above the wave-3 high. Stop: wave-4 low minus the ATR buffer.
  Target: wave-4 low + wave 1 (wave 5 ≈ wave 1).

A position is closed at the stop, at the target, when the next swing high is
confirmed (the wave has topped), or when the structure turns bearish.
Each setup is traded at most once. Long-only.
"""

from __future__ import annotations

import numpy as np

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.analysis import elliott as ew
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.elliott import ElliottWaves, detect, to_numpy


class ElliottWaveTrader(BaseAlgorithm):
    name = "elliott_wave"
    description = "Trade Elliott wave 3 (and 5) breakouts with wave-based stops and Fibonacci targets"
    vectorized = False
    params = {
        "atr_period": IntParam(14, 5, 50, "ATR period for swing size and stop buffer"),
        "swing_atr": Param(4.0, 1.5, 10.0, "minimum swing size in ATRs (zigzag sensitivity)"),
        "wave2_min": Param(0.382, 0.1, 0.618, "minimum wave-2 retracement of wave 1"),
        "wave2_max": Param(0.786, 0.5, 0.99, "maximum wave-2 retracement of wave 1"),
        "target_ext": Param(1.618, 1.0, 3.0, "wave-3 target as multiple of wave 1"),
        "stop_atr": Param(0.5, 0.0, 3.0, "stop buffer below the invalidation level, in ATRs"),
        "min_score": Param(0.3, 0.0, 0.9, "minimum Fibonacci conformity score of the setup"),
        "trade_wave5": IntParam(1, 0, 1, "also trade wave-5 breakouts"),
    }

    def signals(self, o, p):
        o = to_numpy(o)
        high, low, close = (np.asarray(a, float) for a in (o.high, o.low, o.close))
        n = len(close)
        pivots, atr = detect(o, int(p["atr_period"]), float(p["swing_atr"]))
        ws = ew.wave_states(pivots, n)
        # plain Python floats: this loop runs once per optimizer candidate
        high, low, close, atr = high.tolist(), low.tolist(), close.tolist(), atr.tolist()
        states = ws.state.tolist()
        pos = [0.0] * n
        holding = False
        stop = target = 0.0
        entry_t = -1
        setup = None  # (kind, pivots) of the setup currently armed
        traded: set[int] = set()  # setups already used (keyed by the bar they appeared)
        setup_t = -1
        last_pivot_t = {piv.confirmed: piv for piv in pivots}

        for t in range(n):
            if t in ws.setups:  # pivots changed at this bar -> re-evaluate the armed setup
                seq, score = ws.setups[t]
                state = states[t]
                setup, setup_t = None, t
                if state == ew.STATE_WAVE3 and score >= p["min_score"]:
                    w1 = seq[1].price - seq[0].price
                    retrace = (seq[1].price - seq[2].price) / w1
                    if p["wave2_min"] <= retrace <= p["wave2_max"]:
                        setup = ("w3", seq)
                elif state == ew.STATE_WAVE5 and p["trade_wave5"] and score >= p["min_score"]:
                    setup = ("w5", seq)
            if holding:
                new_top = t in last_pivot_t and last_pivot_t[t].kind == 1 and last_pivot_t[t].index > entry_t
                if low[t] <= stop or high[t] >= target or new_top or states[t] < 0:
                    holding = False
            elif setup is not None and setup_t not in traded and atr[t] == atr[t]:  # ATR warmed up (not NaN)
                kind, seq = setup
                trigger = seq[1].price if kind == "w3" else seq[3].price
                if close[t] > trigger:
                    w1 = seq[1].price - seq[0].price
                    base = seq[2].price if kind == "w3" else seq[4].price
                    stop = base - p["stop_atr"] * atr[t]
                    target = base + (p["target_ext"] if kind == "w3" else 1.0) * w1
                    if close[t] < target:
                        holding, entry_t = True, t
                        traded.add(setup_t)
            pos[t] = 1.0 if holding else 0.0
        return np.asarray(pos)

    def chart_indicators(self):
        return [ElliottWaves(atr_period=self.values["atr_period"], swing_atr=self.values["swing_atr"])]
