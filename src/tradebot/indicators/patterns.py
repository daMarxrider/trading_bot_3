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

#: marker codes of the ``pattern`` outputs: positive bullish, negative bearish, 10+ neutral. The first of a
#: dict wins when several match on one bar, so the stronger patterns come first
CANDLE_CODES = {"morning_star": 3, "three_white_soldiers": 4, "bull_engulfing": 2, "piercing": 6, "hammer": 1,
                "bull_harami": 5, "tweezer_bottom": 7,
                "evening_star": -3, "three_black_crows": -4, "bear_engulfing": -2, "dark_cloud": -6,
                "shooting_star": -1, "bear_harami": -5, "tweezer_top": -7}
#: doji is an alert only: it is frequent and neutral, so marking every one clutters the chart. The texts pair the
#: colour (up / down) with a triangle.
UP, DOWN = "\u25b2", "\u25bc"
CANDLE_LABELS = {1: f"{UP} Ham", 2: f"{UP} Eng", 3: f"{UP} MStar", 4: f"{UP} 3WS", 5: f"{UP} Har", 6: f"{UP} Pierce",
                 7: f"{UP} Twz", -1: f"{DOWN} Star", -2: f"{DOWN} Eng", -3: f"{DOWN} EStar", -4: f"{DOWN} 3BC",
                 -5: f"{DOWN} Har", -6: f"{DOWN} Cloud", -7: f"{DOWN} Twz"}
CANDLE_HELP = {1: "Hammer: long lower shadow after a fall",
               2: "Bullish engulfing: green body covers the previous red one",
               3: "Morning star: long red, small body, green closing past its middle",
               4: "Three white soldiers: three long green candles, each closing higher",
               5: "Bullish harami: small green body inside the previous long red one, after a fall",
               6: "Piercing line: green candle opening below the previous low, closing past its middle",
               7: "Tweezer bottom: red then green candle with the same low, after a fall",
               -1: "Shooting star: long upper shadow after a rise",
               -2: "Bearish engulfing: red body covers the previous green one",
               -3: "Evening star: long green, small body, red closing past its middle",
               -4: "Three black crows: three long red candles, each closing lower",
               -5: "Bearish harami: small red body inside the previous long green one, after a rise",
               -6: "Dark cloud cover: red candle opening above the previous high, closing past its middle",
               -7: "Tweezer top: green then red candle with the same high, after a rise"}
CHART_CODES = {"double_bottom": 1, "inverse_head_shoulders": 2, "cup_handle": 5, "ascending_triangle": 3,
               "falling_wedge": 6, "bull_pennant": 7, "bull_flag": 4,
               "double_top": -1, "head_shoulders": -2, "descending_triangle": -3, "rising_wedge": -6,
               "bear_pennant": -7, "bear_flag": -4, "symmetrical_triangle": 10}
CHART_LABELS = {1: f"{UP} W", 2: f"{UP} iH&S", 3: f"{UP} Asc", 4: f"{UP} Flag", 5: f"{UP} Cup", 6: f"{UP} Wedge",
                7: f"{UP} Pennant", -1: f"{DOWN} M", -2: f"{DOWN} H&S", -3: f"{DOWN} Desc", -4: f"{DOWN} Flag",
                -6: f"{DOWN} Wedge", -7: f"{DOWN} Pennant", 10: "\u25c6 Sym"}
CHART_HELP = {1: "Double bottom (W)", 2: "Inverse head and shoulders",
              3: "Ascending triangle: flat top, rising lows", 4: "Bull flag: sharp rise, then a shallow pullback",
              5: "Cup and handle: rounded bottom, then a small dip",
              6: "Falling wedge: both edges fall and converge",
              7: "Bull pennant: sharp rise, then a narrowing range",
              -1: "Double top (M)", -2: "Head and shoulders",
              -3: "Descending triangle: flat bottom, falling highs",
              -4: "Bear flag: sharp fall, then a shallow bounce",
              -6: "Rising wedge: both edges rise and converge",
              -7: "Bear pennant: sharp fall, then a narrowing range",
              10: "Symmetrical triangle: falling highs, rising lows (no direction)"}


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
    """Candlestick patterns: doji, hammer, shooting star, engulfing, morning/evening star, three soldiers/crows,
    harami, piercing/dark cloud, tweezers.

    ``pattern`` marks the candles on the chart (positive codes bullish, negative bearish); every pattern is also
    an alert (``candles.hammer``) and so are ``bullish`` / ``bearish`` (any bullish / bearish pattern).
    """

    name = "candles"
    description = "Candlestick patterns (hammer, engulfing, stars, three soldiers/crows, harami, tweezers) with markers and alerts"
    params = {"doji_ratio": Param(0.1, 0.01, 0.4, "largest body, as a share of the candle's range, of a doji"),
              "trend": IntParam(5, 1, 50, "candles a hammer / shooting star must follow a fall / rise over"),
              "long_ratio": Param(0.6, 0.3, 0.95, "share of its range the first candle of a star must be body")}
    outputs = ("pattern", "anchor", "bullish", "bearish", *pt.CANDLE_PATTERNS)
    labels = {"pattern": CANDLE_LABELS}
    label_help = {"pattern": CANDLE_HELP}
    label_signs = True
    label_anchor = "anchor"
    hidden = ("anchor",)
    alerts = {"bullish": "any bullish candlestick pattern (hammer, bullish engulfing, morning star, three white soldiers, harami, piercing, tweezer bottom)",
              "bearish": "any bearish candlestick pattern (shooting star, bearish engulfing, evening star, three black crows, harami, dark cloud, tweezer top)",
              "doji": "doji (open and close almost equal)",
              "hammer": "hammer (long lower shadow after a fall)",
              "shooting_star": "shooting star (long upper shadow after a rise)",
              "bull_engulfing": "bullish engulfing (green body covers the previous red one)",
              "bear_engulfing": "bearish engulfing (red body covers the previous green one)",
              "morning_star": "morning star (long red, small body, green closing past its middle)",
              "evening_star": "evening star (long green, small body, red closing past its middle)",
              "three_white_soldiers": "three white soldiers (three long green candles, each closing higher)",
              "three_black_crows": "three black crows (three long red candles, each closing lower)",
              "bull_harami": "bullish harami (small green body inside the previous long red one, after a fall)",
              "bear_harami": "bearish harami (small red body inside the previous long green one, after a rise)",
              "piercing": "piercing line (green candle opening below the previous low, closing past its middle)",
              "dark_cloud": "dark cloud cover (red candle opening above the previous high, closing past its middle)",
              "tweezer_bottom": "tweezer bottom (red then green candle with the same low, after a fall)",
              "tweezer_top": "tweezer top (green then red candle with the same high, after a rise)"}

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
    """Chart patterns from swing points: double top/bottom, head and shoulders (and inverse), triangles, wedges,
    flags, pennants, cup and handle.

    A pattern is reported on the bar its last swing point is confirmed, ``strength`` candles after it formed, never
    earlier and never changed afterwards (``pattern`` marks that bar). ``bullish`` / ``bearish`` are true on the
    bar of any bullish / bearish pattern; the symmetrical triangle is neutral and only has its own alert.
    """

    name = "chart_patterns"
    description = "Chart patterns (double top/bottom, head and shoulders, triangles, wedges, flags, cup) with markers and alerts"
    params = {"strength": IntParam(5, 2, 30, "candles on each side that make a swing point"),
              "tolerance": Param(0.03, 0.005, 0.1, "relative difference still counted as equal (tops, shoulders)"),
              "depth": Param(0.03, 0.005, 0.2, "least relative dip between two tops / rise between two bottoms")}
    outputs = ("pattern", "anchor", "bullish", "bearish", *pt.CHART_PATTERNS)
    labels = {"pattern": CHART_LABELS}
    label_anchor = "anchor"
    hidden = ("anchor",)
    label_help = {"pattern": CHART_HELP}
    label_signs = True
    alerts = {"bullish": "any bullish chart pattern (double bottom, inverse H&S, ascending triangle, falling wedge, "
                         "bull flag / pennant, cup and handle)",
              "bearish": "any bearish chart pattern (double top, H&S, descending triangle, rising wedge, "
                         "bear flag / pennant)",
              "double_top": "double top confirmed",
              "double_bottom": "double bottom confirmed",
              "head_shoulders": "head and shoulders confirmed",
              "inverse_head_shoulders": "inverse head and shoulders confirmed",
              "ascending_triangle": "ascending triangle (flat top, rising lows)",
              "descending_triangle": "descending triangle (flat bottom, falling highs)",
              "symmetrical_triangle": "symmetrical triangle (falling highs, rising lows)",
              "bull_flag": "bull flag (sharp rise, then a shallow pullback)",
              "bear_flag": "bear flag (sharp fall, then a shallow bounce)",
              "rising_wedge": "rising wedge (both edges rise and converge, bearish)",
              "falling_wedge": "falling wedge (both edges fall and converge, bullish)",
              "bull_pennant": "bull pennant (sharp rise, then a narrowing range)",
              "bear_pennant": "bear pennant (sharp fall, then a narrowing range)",
              "cup_handle": "cup and handle (rounded bottom between two level rims, then a small dip)"}

    @classmethod
    def compute(cls, o, strength, tolerance, depth):
        n = len(o)
        a = to_numpy(o)
        f = pt.chart_patterns(a.high, a.low, a.close, int(strength), tolerance, depth)
        warm = np.arange(n) < 2 * int(strength) + 1
        bull = np.any([f[k] for k in pt.BULLISH_CHART], axis=0)
        bear = np.any([f[k] for k in pt.BEARISH_CHART], axis=0)
        code = _code(f, CHART_CODES, n)
        # bullish markers sit under the candle's low, the others over its high
        anchor = np.where(code > 0, np.asarray(a.low, float), np.asarray(a.high, float))
        out = {"pattern": np.where(warm, np.nan, code), "anchor": anchor,
               "bullish": _flag(bull, warm), "bearish": _flag(bear, warm)}
        out.update({k: _flag(f[k], warm) for k in pt.CHART_PATTERNS})
        return _out(o, **out)


class SupportResistance(BaseIndicator):
    """Automatic support and resistance levels from swing points, confirmed by repeated touches or volume.

    ``support`` / ``resistance`` are the nearest known levels below / above the close, drawn as horizontal bands
    (``zone`` wide) that start where the level changes. Only swing points confirmed by that bar are used, so a level never appears in hindsight.
    """

    name = "support_resistance"
    description = "Automatic support and resistance levels from swing points and volume"
    params = {"strength": IntParam(5, 2, 30, "candles on each side that make a swing point"),
              "zone": Param(0.01, 0.001, 0.05, "swing points this close (relative) form one level"),
              "min_touches": IntParam(2, 1, 6, "swing points a level needs (or one on a high-volume candle)"),
              "lookback": IntParam(250, 20, 2000, "candles back that swing points still count"),
              "proximity": Param(0.01, 0.001, 0.05, "how close (relative) the close must be for the near alerts")}
    outputs = ("support", "resistance", "near_support", "near_resistance", "breakout_up", "breakout_down")
    zones = {"support": ("up", "S"), "resistance": ("down", "R")}
    zones = {"support": ("up", "S"), "resistance": ("down", "R")}
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


class VolumeLevels(BaseIndicator):
    """Support and resistance at the high-volume nodes of the volume profile of the last ``lookback`` candles.

    Where much was traded, price tends to stall: ``support`` / ``resistance`` are the nearest such nodes below /
    above the close (bands), ``poc`` the busiest price (point of control). The profile is rebuilt every
    ``lookback / 10`` candles from the candles up to then only, so levels hold still in between and never use later
    candles. Needs volume data (NaN without).
    """

    name = "volume_levels"
    description = "Support and resistance at the high-volume nodes of the volume profile"
    params = {"lookback": IntParam(250, 20, 2000, "candles the volume profile covers"),
              "bins": IntParam(40, 10, 200, "price bins of the profile"),
              "node_mult": Param(1.5, 1.0, 5.0, "volume a node needs, as a multiple of the average bin"),
              "zone": Param(0.006, 0.001, 0.05, "height (relative) of the drawn band and of the near alerts"),
              }
    outputs = ("support", "resistance", "poc", "near_support", "near_resistance", "breakout_up", "breakout_down")
    uses_volume = True
    zones = {"support": ("up", "S"), "resistance": ("down", "R"), "poc": ("muted", "POC")}
    #: drawn as bands only; the levels move with the profile, so ``evolve`` is not offered them as numbers (the
    #: alerts below are the way to use them in rules)
    hidden = ("support", "resistance", "poc")
    alerts = {"near_support": "the close is inside the band of the support node",
              "near_resistance": "the close is inside the band of the resistance node",
              "breakout_up": "the close crosses above the resistance node that was valid the bar before",
              "breakout_down": "the close crosses below the support node that was valid the bar before"}

    @classmethod
    def compute(cls, o, lookback, bins, node_mult, zone):
        n = len(o)
        a = to_numpy(o)
        c = np.asarray(a.close, float)
        sup, res, poc = pt.volume_levels(a.high, a.low, c, a.volume, int(lookback), int(bins), node_mult)
        warm = np.arange(n) < 2
        ps, pr, pc = pt._prev(sup), pt._prev(res), pt._prev(c)
        with np.errstate(invalid="ignore"):
            near_s = c <= sup * (1 + zone)
            near_r = c >= res * (1 - zone)
            up = (c > pr) & (pc <= pr)
            down = (c < ps) & (pc >= ps)
        return _out(o, support=sup, resistance=res, poc=poc, near_support=_flag(near_s, warm),
                    near_resistance=_flag(near_r, warm), breakout_up=_flag(up, warm),
                    breakout_down=_flag(down, warm))
