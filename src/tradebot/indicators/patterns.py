# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Pattern detection indicators (#164): candlestick patterns, chart patterns, support/resistance.

All three only look at the current and earlier candles (see :mod:`tradebot.analysis.patterns`), so they work in the
Scanner, as combinator conditions (``candles.bull_engulfing & rsi(14) < 40``, ``chart_patterns.double_bottom``,
``support_resistance.breakout_up``) and in ``evolve``. The ``pattern`` outputs are drawn as text markers on the
chart; the alerts are 0/1 per bar.
"""

from __future__ import annotations

import numpy as np

from tradebot.analysis import patterns as pt
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.base import BaseIndicator
from tradebot.indicators.elliott import to_numpy

#: marker codes of the ``pattern`` outputs: positive bullish, negative bearish, 10+ neutral
CANDLE_CODES = {"hammer": 1, "bull_engulfing": 2, "morning_star": 3, "shooting_star": -1, "bear_engulfing": -2,
                "evening_star": -3}
#: doji is an alert only: it is frequent and neutral, so marking every one clutters the chart
CANDLE_LABELS = {1: "Ham", 2: "Eng+", 3: "MStar", -1: "Star", -2: "Eng-", -3: "EStar"}
CHART_CODES = {"double_bottom": 1, "inverse_head_shoulders": 2, "ascending_triangle": 3, "bull_flag": 4,
               "double_top": -1, "head_shoulders": -2, "descending_triangle": -3, "bear_flag": -4,
               "symmetrical_triangle": 10}
CHART_LABELS = {1: "W", 2: "iH&S", 3: "Asc.Tri", 4: "Bull flag", -1: "M", -2: "H&S", -3: "Desc.Tri",
                -4: "Bear flag", 10: "Sym.Tri"}


def _out(like, **arrays) -> dict:
    return {k: ops.asarray(v, like.close) for k, v in arrays.items()}


def _flag(cond, warm: np.ndarray) -> np.ndarray:
    return np.where(warm, np.nan, np.asarray(cond, dtype=float))


def _code(found: dict[str, np.ndarray], codes: dict[str, int], n: int) -> np.ndarray:
    """Marker code per bar; where several patterns match the first of ``codes`` wins."""
    code = np.full(n, np.nan)
    for name in reversed(list(codes)):
        code[found[name]] = codes[name]
    return code


class Candles(BaseIndicator):
    """Candlestick patterns: doji, hammer, shooting star, engulfing, morning/evening star.

    ``pattern`` marks the candles on the chart (positive codes bullish, negative bearish); every pattern is also
    an alert (``candles.hammer``) and so are ``bullish`` / ``bearish`` (any bullish / bearish pattern).
    """

    name = "candles"
    description = "Candlestick patterns (doji, hammer, engulfing, morning/evening star) with chart markers and alerts"
    params = {"doji_ratio": Param(0.1, 0.01, 0.4, "largest body, as a share of the candle's range, of a doji"),
              "trend": IntParam(5, 1, 50, "candles a hammer / shooting star must follow a fall / rise over"),
              "long_ratio": Param(0.6, 0.3, 0.95, "share of its range the first candle of a star must be body")}
    outputs = ("pattern", "anchor", "bullish", "bearish", "doji", "hammer", "shooting_star", "bull_engulfing",
               "bear_engulfing", "morning_star", "evening_star")
    labels = {"pattern": CANDLE_LABELS}
    label_anchor = "anchor"
    hidden = ("anchor",)
    alerts = {"bullish": "any bullish candlestick pattern (hammer, bullish engulfing, morning star)",
              "bearish": "any bearish candlestick pattern (shooting star, bearish engulfing, evening star)",
              "doji": "doji (open and close almost equal)",
              "hammer": "hammer (long lower shadow after a fall)",
              "shooting_star": "shooting star (long upper shadow after a rise)",
              "bull_engulfing": "bullish engulfing (green body covers the previous red one)",
              "bear_engulfing": "bearish engulfing (red body covers the previous green one)",
              "morning_star": "morning star (long red, small body, green closing past its middle)",
              "evening_star": "evening star (long green, small body, red closing past its middle)"}

    @classmethod
    def compute(cls, o, doji_ratio, trend, long_ratio):
        n = len(o)
        a = to_numpy(o)
        f = pt.candlestick(a.open, a.high, a.low, a.close, doji_ratio, int(trend), long_ratio)
        warm = f["warmup"]
        bull = np.any([f[k] for k in pt.BULLISH_CANDLES], axis=0)
        bear = np.any([f[k] for k in pt.BEARISH_CANDLES], axis=0)
        code = _code(f, CANDLE_CODES, n)
        # bullish markers sit under the candle's low, the others over its high
        anchor = np.where(code > 0, np.asarray(a.low, float), np.asarray(a.high, float))
        out = {"pattern": np.where(warm, np.nan, code), "anchor": anchor,
               "bullish": _flag(bull, warm), "bearish": _flag(bear, warm)}
        out.update({k: _flag(f[k], warm) for k in pt.CANDLE_PATTERNS})
        return _out(o, **out)


class ChartPatterns(BaseIndicator):
    """Chart patterns from swing points: double top/bottom, head and shoulders (and inverse), triangles, flags.

    A pattern is reported on the bar its last swing point is confirmed, ``strength`` candles after it formed, never
    earlier and never changed afterwards (``pattern`` marks that bar). ``bullish`` / ``bearish`` are true on the
    bar of any bullish / bearish pattern; the symmetrical triangle is neutral and only has its own alert.
    """

    name = "chart_patterns"
    description = "Chart patterns (double top/bottom, head and shoulders, triangles, flags) with markers and alerts"
    params = {"strength": IntParam(5, 2, 30, "candles on each side that make a swing point"),
              "tolerance": Param(0.03, 0.005, 0.1, "relative difference still counted as equal (tops, shoulders)"),
              "depth": Param(0.03, 0.005, 0.2, "least relative dip between two tops / rise between two bottoms")}
    outputs = ("pattern", "bullish", "bearish", *pt.CHART_PATTERNS)
    labels = {"pattern": CHART_LABELS}
    alerts = {"bullish": "any bullish chart pattern (double bottom, inverse H&S, ascending triangle, bull flag)",
              "bearish": "any bearish chart pattern (double top, H&S, descending triangle, bear flag)",
              "double_top": "double top confirmed",
              "double_bottom": "double bottom confirmed",
              "head_shoulders": "head and shoulders confirmed",
              "inverse_head_shoulders": "inverse head and shoulders confirmed",
              "ascending_triangle": "ascending triangle (flat top, rising lows)",
              "descending_triangle": "descending triangle (flat bottom, falling highs)",
              "symmetrical_triangle": "symmetrical triangle (falling highs, rising lows)",
              "bull_flag": "bull flag (sharp rise, then a shallow pullback)",
              "bear_flag": "bear flag (sharp fall, then a shallow bounce)"}

    @classmethod
    def compute(cls, o, strength, tolerance, depth):
        n = len(o)
        a = to_numpy(o)
        f = pt.chart_patterns(a.high, a.low, a.close, int(strength), tolerance, depth)
        warm = np.arange(n) < 2 * int(strength) + 1
        bull = np.any([f[k] for k in pt.BULLISH_CHART], axis=0)
        bear = np.any([f[k] for k in pt.BEARISH_CHART], axis=0)
        out = {"pattern": np.where(warm, np.nan, _code(f, CHART_CODES, n)),
               "bullish": _flag(bull, warm), "bearish": _flag(bear, warm)}
        out.update({k: _flag(f[k], warm) for k in pt.CHART_PATTERNS})
        return _out(o, **out)


class SupportResistance(BaseIndicator):
    """Automatic support and resistance levels from swing points, confirmed by repeated touches or volume.

    ``support`` / ``resistance`` are the nearest known levels below / above the close, drawn as lines that step as
    levels are found. Only swing points confirmed by that bar are used, so a level never appears in hindsight.
    """

    name = "support_resistance"
    description = "Automatic support and resistance levels from swing points and volume"
    params = {"strength": IntParam(5, 2, 30, "candles on each side that make a swing point"),
              "zone": Param(0.01, 0.001, 0.05, "swing points this close (relative) form one level"),
              "min_touches": IntParam(2, 1, 6, "swing points a level needs (or one on a high-volume candle)"),
              "lookback": IntParam(250, 20, 2000, "candles back that swing points still count"),
              "proximity": Param(0.01, 0.001, 0.05, "how close (relative) the close must be for the near alerts")}
    outputs = ("support", "resistance", "near_support", "near_resistance", "breakout_up", "breakout_down")
    alerts = {"near_support": "the close is within the proximity of the support level",
              "near_resistance": "the close is within the proximity of the resistance level",
              "breakout_up": "the close crosses above the resistance that was valid the bar before",
              "breakout_down": "the close crosses below the support that was valid the bar before"}

    @classmethod
    def compute(cls, o, strength, zone, min_touches, lookback, proximity):
        n = len(o)
        a = to_numpy(o)
        c = np.asarray(a.close, float)
        sup, res = pt.support_resistance(a.high, a.low, c, a.volume, int(strength), zone, int(min_touches),
                                         int(lookback))
        warm = np.arange(n) < 2 * int(strength) + 1
        ps, pr = pt._prev(sup), pt._prev(res)
        pc = pt._prev(c)
        with np.errstate(invalid="ignore"):
            near_s = c <= sup * (1 + proximity)
            near_r = c >= res * (1 - proximity)
            up = (c > pr) & (pc <= pr)
            down = (c < ps) & (pc >= ps)
        return _out(o, support=sup, resistance=res, near_support=_flag(near_s, warm),
                    near_resistance=_flag(near_r, warm), breakout_up=_flag(up, warm),
                    breakout_down=_flag(down, warm))
