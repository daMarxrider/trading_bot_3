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
                   "evening_star", "three_white_soldiers", "three_black_crows", "bull_harami", "bear_harami",
                   "piercing", "dark_cloud", "tweezer_bottom", "tweezer_top")
BULLISH_CANDLES = ("hammer", "bull_engulfing", "morning_star", "three_white_soldiers", "bull_harami", "piercing",
                   "tweezer_bottom")
BEARISH_CANDLES = ("shooting_star", "bear_engulfing", "evening_star", "three_black_crows", "bear_harami",
                   "dark_cloud", "tweezer_top")

CHART_PATTERNS = ("double_top", "double_bottom", "head_shoulders", "inverse_head_shoulders", "ascending_triangle",
                  "descending_triangle", "symmetrical_triangle", "bull_flag", "bear_flag", "rising_wedge", "falling_wedge", "bull_pennant",
                  "bear_pennant", "cup_handle")
BULLISH_CHART = ("double_bottom", "inverse_head_shoulders", "ascending_triangle", "bull_flag", "falling_wedge",
                 "bull_pennant", "cup_handle")
BEARISH_CHART = ("double_top", "head_shoulders", "descending_triangle", "bear_flag", "rising_wedge", "bear_pennant")


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
    * engulfing: the body (at least half of the candle's range) covers the previous, opposite coloured body
    * morning / evening star: a long (``long_ratio`` of the range) candle, a small body, then a candle closing past
      the middle of the first one, in the opposite direction
    * three white soldiers / three black crows: three long candles in a row, each opening inside the previous body
      and closing beyond its close
    * harami: a small body inside the previous long, opposite coloured body, after a fall (bullish) / rise (bearish)
    * piercing / dark cloud: after a fall / rise, a candle opening beyond the previous long candle's extreme and
      closing past the middle of its body, but not past its open
    * tweezer bottom / top: two candles of opposite colour whose lows / highs almost match, after a fall / rise
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
        strong = body >= 0.5 * rng  # a real body, not a candle that is mostly shadow
        bull_eng = (pc < po) & (c > o) & (o <= pc) & (c >= po) & (body > pb) & strong
        bear_eng = (pc > po) & (c < o) & (o >= pc) & (c <= po) & (body > pb) & strong
        o2, c2, b2, r2 = _prev(o, 2), _prev(c, 2), _prev(body, 2), _prev(rng, 2)
        small = pb <= 0.5 * b2
        long2 = (b2 >= long_ratio * r2) & (r2 > 0)
        mid2 = (o2 + c2) / 2
        morning = long2 & (c2 < o2) & small & (np.maximum(po, pc) <= c2) & (c > o) & (c > mid2)
        evening = long2 & (c2 > o2) & small & (np.minimum(po, pc) >= c2) & (c < o) & (c < mid2)
        ph, pl, pr = _prev(h), _prev(lo), _prev(rng)
        o3, c3, r3, b3 = _prev(o, 2), _prev(c, 2), _prev(rng, 2), _prev(body, 2)
        pup, pdown = _prev(c) > _prev(o), _prev(c) < _prev(o)
        pbig = (pb >= long_ratio * pr) & (pr > 0)
        big = (body >= long_ratio * rng) & ok
        soldiers = (c > o) & pup & (c3 > o3) & (c > pc) & (pc > c3) & (o > po) & (o < pc) & (po > o3) \
            & (po < c3) & big & pbig & (b3 >= long_ratio * r3) & (r3 > 0) & (upper <= 0.3 * body)
        crows = (c < o) & pdown & (c3 < o3) & (c < pc) & (pc < c3) & (o < po) & (o > pc) & (po < o3) & (po > c3) & big & pbig \
            & (b3 >= long_ratio * r3) & (r3 > 0) & (c3 < o3) & (lower <= 0.3 * body)
        inside = (body < pb) & (np.maximum(o, c) <= np.maximum(po, pc)) & (np.minimum(o, c) >= np.minimum(po, pc))
        bull_har = pbig & pdown & (c > o) & inside & down
        bear_har = pbig & pup & (c < o) & inside & up
        pmid = (po + pc) / 2
        piercing = pbig & pdown & (c > o) & (o < pl) & (c > pmid) & (c < po) & down
        cloud = pbig & pup & (c < o) & (o > ph) & (c < pmid) & (c > po) & up
        near = np.maximum(rng, pr) * 0.1
        twz_bot = pdown & (c > o) & (np.abs(lo - pl) <= near) & down & (pr > 0) & ok & (pb >= 0.3 * pr)
        twz_top = pup & (c < o) & (np.abs(h - ph) <= near) & up & (pr > 0) & ok & (pb >= 0.3 * pr)
    out = {"doji": doji, "hammer": hammer, "shooting_star": star, "bull_engulfing": bull_eng,
           "bear_engulfing": bear_eng, "morning_star": morning, "evening_star": evening,
           "three_white_soldiers": soldiers, "three_black_crows": crows, "bull_harami": bull_har,
           "bear_harami": bear_har, "piercing": piercing, "dark_cloud": cloud, "tweezer_bottom": twz_bot,
           "tweezer_top": twz_top}
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
                   flag_retrace: float = 0.5, cup_bars: int = 20, cup_depth: float = 0.1, cup_tol: float = 0.05) -> dict[str, np.ndarray]:
    """Boolean arrays per name in :data:`CHART_PATTERNS`, True on the bar the pattern is confirmed.

    Swing-based patterns fire on the bar their last swing point is confirmed; ``tol`` is the relative difference
    still counted as "equal" (tops, shoulders, flat triangle edges) and ``depth`` the relative dip/peak between
    them. Flags and pennants (a ``pole_pct`` move over ``pole_bars`` candles followed by ``flag_bars`` candles moving at most
    ``flag_retrace`` of it back) fire on the first bar the shape is complete; a pennant is a flag whose range narrows. Wedges are triangles
    whose edges slope the same way; the cup and handle needs two level rims at least ``cup_bars`` apart, a U-shaped
    bottom ``cup_depth`` deep and a handle dip of at most half of it, and fires when the handle's low is confirmed.
    A triangle, wedge or cup is marked once per formation.
    """
    h, lo, c = _arrays(high, low, close)
    n = len(c)
    out = {k: np.zeros(n, dtype=bool) for k in CHART_PATTERNS}
    if n == 0:
        return out
    s = max(int(strength), 1)
    highs: list[Pivot] = []
    lows: list[Pivot] = []
    last: dict[str, int] = {}  # bar of the latest mark per swing-based pattern
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
                    name = "ascending_triangle"
                elif lflat and hfall:
                    name = "descending_triangle"
                elif hfall and lrise:
                    name = "symmetrical_triangle"
                elif h2.price > h1.price and l2.price > l1.price and not hflat and not lflat:
                    name = "rising_wedge"  # both edges rise, the lower one faster
                elif h2.price < h1.price and l2.price < l1.price and not hflat and not lflat:
                    name = "falling_wedge"
                else:
                    name = None
                # one mark per formation: not again while its swing points are still in the window
                if name and last.get(name, -1) <= min(h1.index, l1.index):
                    out[name][bar] = True
                    last[name] = bar
        if not p.high and len(highs) >= 2:  # cup and handle: the handle's low is the last swing point
            a, rim = highs[-2], highs[-1]
            cup = [q for q in lows if a.index < q.index < rim.index]
            span = rim.index - a.index
            if cup and rim.index < p.index and span >= cup_bars:
                bottom = min(q.price for q in cup)
                top = min(a.price, rim.price)
                size = top - bottom
                handle = rim.price - p.price
                if _rel(a.price, rim.price) <= cup_tol and size >= cup_depth * top and 0 < handle <= 0.5 * size \
                        and p.index - rim.index <= span // 2 \
                        and np.mean(c[a.index: rim.index + 1] < bottom + 0.25 * size) >= 0.35 \
                        and last.get("cup_handle", -1) <= a.index:
                    out["cup_handle"][bar] = True
                    last["cup_handle"] = bar
    # flags: a sharp pole, then a small counter-move
    span = pole_bars + flag_bars
    for t in range(span, n):
        top, start = c[t - flag_bars], c[t - span]
        fh, fl = h[t - flag_bars + 1: t + 1].max(), lo[t - flag_bars + 1: t + 1].min()
        move = top - start
        if not start > 0:
            continue
        half = flag_bars // 2
        pennant = False
        if half >= 2:  # a pennant's range narrows, a flag's stays about the same
            first = h[t - flag_bars + 1: t - flag_bars + 1 + half].max() - lo[t - flag_bars + 1: t - flag_bars + 1 + half].min()
            second = h[t - half + 1: t + 1].max() - lo[t - half + 1: t + 1].min()
            pennant = second <= 0.6 * first
        if move / start >= pole_pct and top - fl <= flag_retrace * move \
                and fh <= h[t - span: t - flag_bars + 1].max() * (1 + tol):
            out["bull_pennant" if pennant else "bull_flag"][t] = True
        elif -move / start >= pole_pct and fh - top <= flag_retrace * -move \
                and fl >= lo[t - span: t - flag_bars + 1].min() * (1 - tol):
            out["bear_pennant" if pennant else "bear_flag"][t] = True
    for k in ("bull_flag", "bear_flag", "bull_pennant", "bear_pennant"):  # the first bar of a run only
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
        expired = bool(known) and known[0].index < t - lookback  # known is in bar order
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


def volume_levels(high, low, close, volume, lookback: int = 250, bins: int = 40, node_mult: float = 1.5,
                  refresh: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Support, resistance and point of control from the volume profile of the last ``lookback`` candles.

    A candle's volume goes into the price bin of its typical price; the bins split the window's low-to-high range.
    A bin that is the busiest within its neighbours and holds at least ``node_mult`` times the average bin volume
    is a high-volume node, a level where much was traded. Support is the highest node at or below the close,
    resistance the lowest above it, the point of control (POC) the busiest bin. The profile is rebuilt every
    ``refresh`` candles (default a tenth of ``lookback``) from the candles up to that bar, so levels hold still
    between rebuilds and never use later candles. NaN without volume.
    """
    h, lo, c = _arrays(high, low, close)
    n = len(c)
    sup, res, poc = (np.full(n, np.nan) for _ in range(3))
    if n == 0 or volume is None:
        return sup, res, poc
    v = np.asarray(volume, dtype=float)
    tp = (h + lo + c) / 3
    lookback, bins = max(int(lookback), 2), max(int(bins), 3)
    step = int(refresh) if refresh > 0 else max(lookback // 10, 1)
    nodes = np.empty(0)
    centre = np.nan
    for t in range(n):
        if t % step == 0 and t >= 1:
            a = max(0, t - lookback + 1)
            w = slice(a, t + 1)
            ok = np.isfinite(tp[w]) & np.isfinite(v[w]) & np.isfinite(h[w]) & np.isfinite(lo[w])
            lo_p, hi_p = (np.min(lo[w][ok]), np.max(h[w][ok])) if ok.any() else (np.nan, np.nan)
            nodes, centre = np.empty(0), np.nan
            if ok.any() and hi_p > lo_p and v[w][ok].sum() > 0:
                width = (hi_p - lo_p) / bins
                idx = np.clip(((tp[w][ok] - lo_p) / width).astype(int), 0, bins - 1)
                hist = np.bincount(idx, weights=v[w][ok], minlength=bins)
                padded = np.r_[0.0, hist, 0.0]
                peak = (hist > padded[:-2]) & (hist >= padded[2:]) & (hist >= node_mult * hist.mean())
                nodes = lo_p + (np.flatnonzero(peak) + 0.5) * width
                centre = lo_p + (hist.argmax() + 0.5) * width
        poc[t] = centre
        if nodes.size and np.isfinite(c[t]):
            below, above = nodes[nodes <= c[t]], nodes[nodes > c[t]]
            sup[t] = below.max() if below.size else np.nan
            res[t] = above.min() if above.size else np.nan
    return sup, res, poc
