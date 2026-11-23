# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Candlestick patterns, chart patterns and support/resistance levels (#164), as pure NumPy functions.

Every function is **causal**: the value on bar ``t`` uses only candles up to ``t``, so signals never change when
later candles arrive (they are safe for algorithms, the Scanner and ``evolve``). Swing points need ``strength``
candles on each side, so a pivot at bar ``p`` is only known (and a pattern built on it only reported) on bar
``p + strength``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CANDLE_PATTERNS = ("doji", "hammer", "shooting_star", "bull_engulfing", "bear_engulfing", "morning_star",
                   "evening_star")
BULLISH_CANDLES = ("hammer", "bull_engulfing", "morning_star")
BEARISH_CANDLES = ("shooting_star", "bear_engulfing", "evening_star")

CHART_PATTERNS = ("double_top", "double_bottom", "head_shoulders", "inverse_head_shoulders", "ascending_triangle",
                  "descending_triangle", "symmetrical_triangle", "bull_flag", "bear_flag")
BULLISH_CHART = ("double_bottom", "inverse_head_shoulders", "ascending_triangle", "bull_flag")
BEARISH_CHART = ("double_top", "head_shoulders", "descending_triangle", "bear_flag")


def _arrays(*a):
    return [np.asarray(x, dtype=float) for x in a]


def _prev(x: np.ndarray, k: int = 1) -> np.ndarray:
    """``x`` shifted ``k`` bars later (NaN at the start)."""
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def candlestick(open_, high, low, close, doji_ratio: float = 0.1, trend: int = 5,
                long_ratio: float = 0.6) -> dict[str, np.ndarray]:
    """Boolean arrays per name in :data:`CANDLE_PATTERNS` plus ``warmup`` (bars too early to judge).

    * doji: body at most ``doji_ratio`` of the candle's range
    * hammer / shooting star: long lower / upper shadow (at least twice the body), almost no opposite shadow, after
      a fall / rise over the last ``trend`` candles
    * engulfing: the body covers the previous, opposite coloured body
    * morning / evening star: a long (``long_ratio`` of the range) candle, a small body, then a candle closing past
      the middle of the first one, in the opposite direction
    """
    o, h, lo, c = _arrays(open_, high, low, close)
    n = len(c)
    body = np.abs(c - o)
    rng = h - lo
    top, bot = np.maximum(o, c), np.minimum(o, c)
    upper, lower = h - top, bot - lo
    ok = rng > 0
    with np.errstate(invalid="ignore"):
        doji = ok & (body <= doji_ratio * rng)
        down = _prev(c) < _prev(c, trend + 1)  # fell before this candle
        up = _prev(c) > _prev(c, trend + 1)
        hammer = ok & (lower >= 2 * body) & (upper <= 0.1 * rng) & (lower > 0) & down
        star = ok & (upper >= 2 * body) & (lower <= 0.1 * rng) & (upper > 0) & up
        po, pc, pb = _prev(o), _prev(c), _prev(body)
        bull_eng = (pc < po) & (c > o) & (o <= pc) & (c >= po) & (body > pb)
        bear_eng = (pc > po) & (c < o) & (o >= pc) & (c <= po) & (body > pb)
        o2, c2, b2, r2 = _prev(o, 2), _prev(c, 2), _prev(body, 2), _prev(rng, 2)
        small = pb <= 0.5 * b2
        long2 = (b2 >= long_ratio * r2) & (r2 > 0)
        mid2 = (o2 + c2) / 2
        morning = long2 & (c2 < o2) & small & (np.maximum(po, pc) <= c2) & (c > o) & (c > mid2)
        evening = long2 & (c2 > o2) & small & (np.minimum(po, pc) >= c2) & (c < o) & (c < mid2)
    out = {"doji": doji, "hammer": hammer, "shooting_star": star, "bull_engulfing": bull_eng,
           "bear_engulfing": bear_eng, "morning_star": morning, "evening_star": evening}
    out = {k: np.asarray(v, dtype=bool) for k, v in out.items()}
    warm = np.zeros(n, dtype=bool)
    warm[: min(n, max(trend + 1, 2) + 1)] = True
    out["warmup"] = warm
    return out


@dataclass(frozen=True)
class Pivot:
    index: int
    high: bool  #: True: swing high, False: swing low
    price: float


def swing_points(high, low, strength: int = 5) -> list[Pivot]:
    """Swing highs/lows: the highest high (lowest low) among ``strength`` candles on each side.

    Ties on the right are allowed, on the left not, so a flat top counts once (its first bar). Sorted by bar;
    the pivot at ``i`` is only known on bar ``i + strength``.
    """
    h, lo = _arrays(high, low)
    n, s = len(h), max(int(strength), 1)
    out = []
    for i in range(s, n - s):
        if np.isnan(h[i]) or np.isnan(lo[i]):
            continue
        if h[i] > h[i - s: i].max() and h[i] >= h[i + 1: i + s + 1].max():
            out.append(Pivot(i, True, float(h[i])))
        if lo[i] < lo[i - s: i].min() and lo[i] <= lo[i + 1: i + s + 1].min():
            out.append(Pivot(i, False, float(lo[i])))
    return out


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def chart_patterns(high, low, close, strength: int = 5, tol: float = 0.03, depth: float = 0.03,
                   triangle_bars: int = 80, pole_bars: int = 10, flag_bars: int = 8, pole_pct: float = 0.08,
                   flag_retrace: float = 0.5) -> dict[str, np.ndarray]:
    """Boolean arrays per name in :data:`CHART_PATTERNS`, True on the bar the pattern is confirmed.

    Swing-based patterns fire on the bar their last swing point is confirmed; ``tol`` is the relative difference
    still counted as "equal" (tops, shoulders, flat triangle edges) and ``depth`` the relative dip/peak between
    them. Flags (a ``pole_pct`` move over ``pole_bars`` candles followed by ``flag_bars`` candles moving at most
    ``flag_retrace`` of it back) fire on the first bar the shape is complete.
    """
    h, lo, c = _arrays(high, low, close)
    n = len(c)
    out = {k: np.zeros(n, dtype=bool) for k in CHART_PATTERNS}
    if n == 0:
        return out
    s = max(int(strength), 1)
    highs: list[Pivot] = []
    lows: list[Pivot] = []
    for p in swing_points(h, lo, s):  # sorted by bar
        bar = p.index + s  # known from here
        (highs if p.high else lows).append(p)
        same, other = (highs, lows) if p.high else (lows, highs)

        def between(a, b, other=other):
            return [q.price for q in other if a.index < q.index < b.index]

        if len(same) >= 2:
            a, b = same[-2], same[-1]
            mid = between(a, b)
            if mid and _rel(a.price, b.price) <= tol:
                dip = (min(a.price, b.price) - min(mid)) if p.high else (max(mid) - max(a.price, b.price))
                if dip >= depth * max(a.price, b.price):
                    out["double_top" if p.high else "double_bottom"][bar] = True
        if len(same) >= 3:
            left, head, right = same[-3], same[-2], same[-1]
            lift = (head.price - max(left.price, right.price)) if p.high \
                else (min(left.price, right.price) - head.price)
            if lift >= depth * head.price and _rel(left.price, right.price) <= tol * 1.5 \
                    and between(left, head) and between(head, right):
                out["head_shoulders" if p.high else "inverse_head_shoulders"][bar] = True
        if len(highs) >= 2 and len(lows) >= 2:
            h1, h2, l1, l2 = highs[-2], highs[-1], lows[-2], lows[-1]
            if min(h1.index, l1.index) >= bar - triangle_bars and (h2.price - l2.price) < (h1.price - l1.price):
                hflat, lflat = _rel(h1.price, h2.price) <= tol, _rel(l1.price, l2.price) <= tol
                hfall, lrise = h2.price < h1.price and not hflat, l2.price > l1.price and not lflat
                if hflat and lrise:
                    out["ascending_triangle"][bar] = True
                elif lflat and hfall:
                    out["descending_triangle"][bar] = True
                elif hfall and lrise:
                    out["symmetrical_triangle"][bar] = True
    # flags: a sharp pole, then a small counter-move
    span = pole_bars + flag_bars
    for t in range(span, n):
        top, start = c[t - flag_bars], c[t - span]
        fh, fl = h[t - flag_bars + 1: t + 1].max(), lo[t - flag_bars + 1: t + 1].min()
        move = top - start
        if not start > 0:
            continue
        if move / start >= pole_pct and top - fl <= flag_retrace * move \
                and fh <= h[t - span: t - flag_bars + 1].max() * (1 + tol):
            out["bull_flag"][t] = True
        elif -move / start >= pole_pct and fh - top <= flag_retrace * -move \
                and fl >= lo[t - span: t - flag_bars + 1].min() * (1 - tol):
            out["bear_flag"][t] = True
    for k in ("bull_flag", "bear_flag"):  # the first bar of a run only
        out[k] = out[k] & ~np.concatenate(([False], out[k][:-1]))
    return out


def support_resistance(high, low, close, volume=None, strength: int = 5, zone: float = 0.01, min_touches: int = 2,
                       lookback: int = 250, volume_mult: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Nearest ``(support, resistance)`` price per bar from the swing points known by then (NaN: none).

    Swing highs and lows within ``zone`` (relative) of each other form one level. A level counts once it has
    ``min_touches`` swing points, or one swing point on a candle with at least ``volume_mult`` times the average
    volume of the 20 candles before it (ignored without volume data). Only swing points from the last ``lookback``
    candles are used. Support is the highest level at or below the close, resistance the lowest above it.
    """
    h, lo, c = _arrays(high, low, close)
    n = len(c)
    sup, res = np.full(n, np.nan), np.full(n, np.nan)
    if n == 0:
        return sup, res
    heavy = np.zeros(n, dtype=bool)
    if volume is not None:
        vol = np.asarray(volume, dtype=float)
        for i in range(1, n):
            base = vol[max(0, i - 20): i]
            if np.isfinite(base).any() and np.isfinite(vol[i]):
                heavy[i] = vol[i] >= volume_mult * np.nanmean(base)
    s = max(int(strength), 1)
    by_bar: dict[int, list[Pivot]] = {}
    for p in swing_points(h, lo, s):
        by_bar.setdefault(p.index + s, []).append(p)
    known: list[Pivot] = []
    levels = np.empty(0)
    for t in range(n):
        expired = any(p.index < t - lookback for p in known)
        if t in by_bar or expired:
            known = [p for p in known + by_bar.get(t, []) if p.index >= t - lookback]
            clusters: list[list] = []  # [mean price, touches, on a heavy-volume candle]
            for p in sorted(known, key=lambda p: p.price):
                if clusters and _rel(clusters[-1][0], p.price) <= zone:
                    m = clusters[-1]
                    m[0] = (m[0] * m[1] + p.price) / (m[1] + 1)
                    m[1] += 1
                    m[2] = m[2] or bool(heavy[p.index])
                else:
                    clusters.append([p.price, 1, bool(heavy[p.index])])
            levels = np.array([m[0] for m in clusters if m[1] >= min_touches or m[2]])
        if levels.size and not np.isnan(c[t]):
            below, above = levels[levels <= c[t]], levels[levels > c[t]]
            sup[t] = below.max() if below.size else np.nan
            res[t] = above.min() if above.size else np.nan
    return sup, res
