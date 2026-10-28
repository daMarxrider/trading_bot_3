# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Automatic Elliott wave detection.

Pipeline:

1. :func:`zigzag` finds swing pivots. A swing has to move at least
   ``swing_atr`` x ATR, so the detector adapts to the asset's volatility and
   timeframe. Each pivot stores the bar where it became *confirmed* (price
   reversed far enough); a pivot is unknown before that bar. Everything that
   feeds trading decisions uses confirmed pivots only, so there is no look-ahead.
2. :func:`impulse_score` / :func:`correction_score` check the hard Elliott
   rules and grade the Fibonacci proportions:

   * impulse 0-1-2-3-4-5: wave 2 retraces < 100 % of wave 1, wave 3 is not the
     shortest of 1/3/5, wave 4 does not enter wave 1's territory
   * correction A-B-C (after an impulse): B does not exceed the impulse end,
     C extends beyond A

   The score (0-1) measures how close the wave ratios are to the textbook
   Fibonacci relations (see ``IMPULSE_RATIOS``).
3. :func:`label_waves` picks the best non-overlapping patterns for drawing
   (a hindsight view of the whole history).
4. :func:`wave_states` is the causal counterpart: for every bar, which wave
   is in progress given the pivots confirmed up to that bar - this is what
   strategies and screeners use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Ideal ratios (Fibonacci) used for scoring; the closest ideal counts.
IMPULSE_RATIOS = {
    "w2/w1": (0.5, 0.618),
    "w3/w1": (1.618, 2.618),
    "w4/w3": (0.236, 0.382),
    "w5/w1": (0.618, 1.0, 1.618),
}
CORRECTION_RATIOS = {"B/A": (0.5, 0.618, 0.786), "C/A": (1.0, 1.618)}
#: Relative deviation from an ideal ratio at which a component scores 0.
RATIO_TOLERANCE = 0.5

# Codes of the ``wave`` labels (hindsight drawing).
LABELS = {0.0: "0", 1.0: "1", 2.0: "2", 3.0: "3", 4.0: "4", 5.0: "5", 11.0: "A", 12.0: "B", 13.0: "C"}

# Causal ``state`` values; negative = the same for a bearish structure.
STATE_NONE = 0
STATE_WAVE3 = 3  #: waves 1-2 done -> wave 3 expected / running
STATE_WAVE4 = 4  #: waves 1-3 done -> wave 4 correction running
STATE_WAVE5 = 5  #: waves 1-4 done -> wave 5 expected / running
STATE_IMPULSE_DONE = 10  #: complete 1-5 impulse -> A-B-C correction expected
STATE_WAVE_A, STATE_WAVE_B, STATE_CORRECTION_DONE = 11, 12, 13


@dataclass(frozen=True)
class Pivot:
    index: int  #: bar of the swing extreme
    price: float
    kind: int  #: +1 swing high, -1 swing low
    confirmed: int  #: first bar at which the pivot is known


@dataclass
class WaveState:
    """Causal wave state per bar plus the pivots that define the current setup."""

    state: np.ndarray  #: int per bar, see STATE_* (sign = direction)
    score: np.ndarray  #: Fibonacci score of the current setup (0-1)
    setups: dict[int, tuple[list[Pivot], float]] = field(default_factory=dict)  #: bar -> (setup pivots, score)


# -- swings ----------------------------------------------------------------------------------------------
def zigzag(high, low, atr, swing_atr: float) -> list[Pivot]:
    """Causal ATR zigzag. Pivots alternate high/low; ``confirmed`` is when the reversal reached the threshold.

    Runs in the optimizer's inner loop, so it works on plain Python floats
    (NumPy scalar indexing is several times slower per bar).
    """
    high = np.asarray(high, float).tolist()
    low = np.asarray(low, float).tolist()
    thresholds = (np.asarray(atr, float) * swing_atr).tolist()
    pivots: list[Pivot] = []
    direction = 0
    hi_i = lo_i = ext_i = -1
    hi = lo = ext = 0.0
    for t, thr in enumerate(thresholds):
        if thr != thr:  # NaN: ATR still warming up
            continue
        h, lw = high[t], low[t]
        if direction == 1:
            if h > ext:
                ext, ext_i = h, t
            elif ext - lw >= thr:
                pivots.append(Pivot(ext_i, ext, 1, t))
                direction, ext, ext_i = -1, lw, t
        elif direction == -1:
            if lw < ext:
                ext, ext_i = lw, t
            elif h - ext >= thr:
                pivots.append(Pivot(ext_i, ext, -1, t))
                direction, ext, ext_i = 1, h, t
        else:
            if hi_i < 0 or h > hi:
                hi, hi_i = h, t
            if lo_i < 0 or lw < lo:
                lo, lo_i = lw, t
            if hi - lo >= thr:
                if lo_i < hi_i:  # rose from the low: the low is the first pivot, now searching a high
                    pivots.append(Pivot(lo_i, lo, -1, t))
                    direction, ext, ext_i = 1, hi, hi_i
                else:
                    pivots.append(Pivot(hi_i, hi, 1, t))
                    direction, ext, ext_i = -1, lo, lo_i
    return pivots


# -- scoring ---------------------------------------------------------------------------------------------
def _closeness(ratio: float, ideals: tuple[float, ...]) -> float:
    dev = min(abs(ratio - i) / i for i in ideals)
    return max(0.0, 1.0 - dev / RATIO_TOLERANCE)


def _oriented(prices, direction: int) -> list[float]:
    """Mirror bearish structures so every rule can be written for the bullish case."""
    return [p * direction for p in prices]


def impulse_score(prices, direction: int = 1) -> float | None:
    """Score a 6-point impulse 0..5 (``direction`` +1 bullish / -1 bearish); None if a rule is broken.

    Fewer points grade a partial impulse (rules and ratios that are already decidable).
    """
    p = _oriented(prices, direction)
    n = len(p)
    if n < 3 or not p[1] > p[0]:
        return None
    w1 = p[1] - p[0]
    w2 = p[1] - p[2]
    if not 0 < w2 < w1:  # wave 2 must retrace, but never beyond the start of wave 1
        return None
    parts = [_closeness(w2 / w1, IMPULSE_RATIOS["w2/w1"])]
    if n >= 4:
        w3 = p[3] - p[2]
        if not p[3] > p[1]:
            return None
        parts.append(_closeness(w3 / w1, IMPULSE_RATIOS["w3/w1"]))
    if n >= 5:
        w4 = p[3] - p[4]
        if not (w4 > 0 and p[4] > p[1]):  # wave 4 must not overlap wave 1
            return None
        parts.append(_closeness(w4 / w3, IMPULSE_RATIOS["w4/w3"]))
    if n >= 6:
        w5 = p[5] - p[4]
        if not p[5] > p[3]:
            return None
        if w3 < w1 and w3 < w5:  # wave 3 is never the shortest
            return None
        parts.append(_closeness(w5 / w1, IMPULSE_RATIOS["w5/w1"]))
    return float(np.mean(parts))


def correction_score(prices, direction: int = 1) -> float | None:
    """Score an A-B-C correction of an impulse in ``direction`` (prices: impulse end, A, B, C)."""
    q = _oriented(prices, direction)
    if len(q) != 4:
        return None
    a = q[0] - q[1]
    b = q[2] - q[1]
    if not (a > 0 and 0 < b and q[2] < q[0] and q[3] < q[1]):
        return None
    c = q[2] - q[3]
    return float(np.mean([_closeness(b / a, CORRECTION_RATIOS["B/A"]), _closeness(c / a, CORRECTION_RATIOS["C/A"])]))


def _direction(first: Pivot) -> int:
    return 1 if first.kind == -1 else -1  # an impulse starting at a low is bullish


# -- hindsight labelling (drawing) -----------------------------------------------------------------------
def label_waves(pivots: list[Pivot], min_score: float = 0.25) -> list[tuple[Pivot, float]]:
    """Best non-overlapping impulses (0-5) and their A-B-C corrections as ``(pivot, label_code)``."""
    candidates = []
    for i in range(len(pivots) - 5):
        seq = pivots[i:i + 6]
        d = _direction(seq[0])
        s = impulse_score([p.price for p in seq], d)
        if s is not None and s >= min_score:
            candidates.append((s, i, d))
    taken: list[tuple[int, int]] = []
    labels: list[tuple[Pivot, float]] = []

    def free(a, b):
        return all(b <= lo or a >= hi for lo, hi in taken)  # shared endpoints are fine

    for s, i, d in sorted(candidates, reverse=True):
        if not free(i, i + 5):
            continue
        taken.append((i, i + 5))
        labels += [(pivots[i + k], float(k)) for k in range(6)]
        corr = pivots[i + 5:i + 9]
        if len(corr) == 4 and free(i + 5, i + 8):
            cs = correction_score([p.price for p in corr], d)
            if cs is not None and cs >= min_score:
                taken.append((i + 5, i + 8))
                labels += [(corr[k], 10.0 + k) for k in (1, 2, 3)]
    return sorted(labels, key=lambda x: x[0].index)


# -- causal state (trading) ------------------------------------------------------------------------------
def classify(confirmed: list[Pivot], impulse_end: int | None) -> tuple[int, list[Pivot], float]:
    """Wave state implied by the confirmed pivots (newest last).

    ``impulse_end`` is the list index of the last completed impulse, used to
    follow the A-B-C correction that comes after it.
    """
    n = len(confirmed)
    if impulse_end is not None and 1 <= n - 1 - impulse_end <= 3:
        k = n - 1 - impulse_end
        d = _direction(confirmed[impulse_end - 5])
        if k < 3:
            return d * (10 + k), confirmed[impulse_end:], 0.0
        cs = correction_score([p.price for p in confirmed[impulse_end:]], d)
        if cs is not None:
            return d * STATE_CORRECTION_DONE, confirmed[impulse_end:], cs
    for size, state in ((6, STATE_IMPULSE_DONE), (5, STATE_WAVE5), (4, STATE_WAVE4), (3, STATE_WAVE3)):
        if n < size:
            continue
        seq = confirmed[-size:]
        d = _direction(seq[0])
        s = impulse_score([p.price for p in seq], d)
        if s is not None:
            return d * state, seq, s
    return STATE_NONE, [], 0.0


def wave_states(pivots: list[Pivot], n_bars: int) -> WaveState:
    """Per-bar causal wave state: the state at bar t only uses pivots confirmed at or before t."""
    state = np.zeros(n_bars, dtype=int)
    score = np.zeros(n_bars)
    setups: dict[int, tuple[list[Pivot], float]] = {}
    confirmed: list[Pivot] = []
    impulse_end = None
    current, current_score = STATE_NONE, 0.0
    order = sorted(pivots, key=lambda p: p.confirmed)
    j = 0
    for t in range(n_bars):
        changed = False
        while j < len(order) and order[j].confirmed <= t:
            confirmed.append(order[j])
            j += 1
            changed = True
        if changed:
            current, seq, current_score = classify(confirmed, impulse_end)
            if abs(current) == STATE_IMPULSE_DONE:
                impulse_end = len(confirmed) - 1
            elif impulse_end is not None and len(confirmed) - 1 - impulse_end > 3:
                impulse_end = None
            setups[t] = (seq, current_score)
        state[t], score[t] = current, current_score
    return WaveState(state, score, setups)
