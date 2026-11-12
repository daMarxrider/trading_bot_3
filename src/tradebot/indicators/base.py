# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Indicator interface.

Writing a new indicator::

    from tradebot.core import ops
    from tradebot.core.registry import IntParam
    from tradebot.indicators.base import BaseIndicator

    class Momentum(BaseIndicator):
        name = "momentum"
        params = {"period": IntParam(10, 2, 100)}
        overlay = False

        @classmethod
        def compute(cls, o, period):
            return {"value": o.close - ops.shift(o.close, period)}

Defining the class registers it; it then shows up in the GUI chart, the
screener, the CLI and can be used by any algorithm.

How each output is drawn can be set with :attr:`BaseIndicator.styles`::

    styles = {"value": Style(color="#ffd166", width=2), "signal": Style(color="accent", dash="dash")}

Outputs without a style (or colour) get the theme's palette colours. ``Style(bars=True)`` draws an output as bars
from 0 that add up when the chart merges candles (volume); without a colour they take the candle's up/down colour.

``compute`` gets the candles as :class:`~tradebot.core.series.OHLC`: ``o.open``, ``o.high``, ``o.low``,
``o.close``, ``o.time`` and ``o.volume`` (#116). Volume is venue-dependent and all NaN for symbols without volume
data (many indices), so an indicator using it sets :attr:`BaseIndicator.uses_volume` and simply returns NaN there.

Alert conditions (#108) are named true/false states per bar, declared with :attr:`BaseIndicator.alerts`
(``{output: description}``). The output is an array of 0/1 (NaN while warming up) that ``compute`` returns like any
other output; alert outputs are always hidden, so the chart doesn't draw them and the evolver doesn't use them as
numbers. The Scanner filters on them and uses the first two as filter templates. Build them with
:func:`tradebot.core.ops.flag`. :attr:`BaseIndicator.screenable` marks indicators that can't be screened.

The strategy evolver (``tradebot evolve``, #83) builds rules from every indicator output whose value on a
bar depends only on that bar and earlier ones (checked automatically), using :attr:`BaseIndicator.units`
to know what an output can be compared with.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from tradebot.core.registry import Param, Registry, resolve_params
from tradebot.core.series import OHLC

INDICATORS: Registry["BaseIndicator"] = Registry("indicator")


@dataclass(frozen=True)
class Style:
    """How the chart draws one indicator output; unset fields keep the chart's default."""

    #: ``"#rrggbb"``, ``"rgba(r, g, b, a)"`` (a in 0..1), or a theme colour that follows theme changes:
    #: ``"up"``, ``"down"``, ``"accent"``, ``"muted"``, ``"text"`` or ``"series[N]"`` (the palette's N-th colour)
    color: str | None = None
    width: float | None = None  #: line width in pixels (default 1.6)
    dash: str | None = None  #: ``None`` (solid), ``"dash"`` or ``"dot"``
    #: bars from 0 instead of a line; merged bars (more candles than pixels) add up, as volume does.
    #: Without a colour, each bar takes its candle's up/down colour.
    bars: bool = False


class BaseIndicator(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    params: ClassVar[dict[str, Param]] = {}
    #: Names of the arrays returned by :meth:`compute`.
    outputs: ClassVar[tuple[str, ...]] = ("value",)
    #: True: drawn on the price pane. False: drawn in its own pane below.
    overlay: ClassVar[bool] = True
    #: Horizontal reference lines for non-overlay panes (e.g. RSI 30/70).
    levels: ClassVar[tuple[float, ...]] = ()
    #: Outputs drawn as text labels instead of lines: ``{output: {code: text}}``; NaN = no label.
    labels: ClassVar[dict[str, dict[float, str]]] = {}
    #: Output whose value positions the labels vertically (default: the close).
    label_anchor: ClassVar[str | None] = None
    #: Outputs that are computed (for algorithms/screeners) but not drawn.
    hidden: ClassVar[tuple[str, ...]] = ()
    #: How outputs are drawn: ``{output: Style(...)}``; outputs without a colour use the theme palette.
    styles: ClassVar[dict[str, Style]] = {}
    #: What each output measures, for the strategy evolver (#83), which only compares like with like:
    #: ``"price"``, ``"osc"`` (0-100), ``"ratio"`` (0.05 = 5 %), ``"dist"`` (a price distance, like the ATR),
    #: ``"volume"`` (an amount traded, like OBV), ``"money"`` (price × volume) or any other name. Unset outputs: ``"price"`` on overlays, ``"osc"`` when all ``levels`` are within 0-100,
    #: otherwise the indicator's own unit (comparable with its other outputs and its ``levels``).
    units: ClassVar[dict[str, str]] = {}

    #: Alert conditions ``{output: description}``; each output is 0/1 per bar (NaN while warming up).
    #: Alert outputs are implicitly hidden. The order matters: the Scanner uses the first two as filter templates.
    alerts: ClassVar[dict[str, str]] = {}
    #: True for indicators that need ``o.volume``: their outputs are NaN on symbols without volume data, and the
    #: chart says so instead of showing an empty pane.
    uses_volume: ClassVar[bool] = False
    #: False for indicators the Scanner must not list (e.g. chart patterns whose labels change with new bars).
    screenable: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        extra = tuple(a for a in cls.alerts if a not in cls.hidden)
        if extra:
            cls.hidden = (*cls.hidden, *extra)
        INDICATORS.register(cls)

    def __init__(self, **params):
        self.values = resolve_params(self.params, params)

    @classmethod
    @abstractmethod
    def compute(cls, o: OHLC, **params) -> dict[str, object]:
        """Return ``{output_name: array}``; arrays share the backend and length of ``o``."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute on a pandas OHLC frame; columns are ``<name>_<output>``."""
        out = self.compute(OHLC.from_frame(df), **self.values)
        return pd.DataFrame({f"{self.name}_{k}": np.asarray(v) for k, v in out.items()}, index=df.index)

    def output_of(self, column: str) -> str:
        """Output name for a column produced by :meth:`__call__` (``"<name>_<output>"``)."""
        return column[len(self.name) + 1:]

    def is_line(self, column: str) -> bool:
        out = self.output_of(column)
        return out not in self.labels and out not in self.hidden

    @classmethod
    def plot_outputs(cls) -> tuple[str, ...]:
        """Outputs that aren't alerts."""
        return tuple(o for o in cls.outputs if o not in cls.alerts)

    @property
    def label(self) -> str:
        args = ",".join(str(v) for v in self.values.values())
        return f"{self.name}({args})" if args else self.name


def is_screenable(cls: type[BaseIndicator]) -> tuple[bool, str]:
    """Whether the Scanner may list ``cls``, and if not, why (for the picker's tooltip).

    Needs ``screenable``, and at least one output the Scanner can show (a visible line or label output) or an alert.
    """
    if not cls.screenable:
        return False, "chart pattern: its values change as new bars come in"
    shown = [o for o in cls.outputs if o not in cls.hidden or o in cls.alerts]
    if not shown and not cls.alerts:
        return False, "no plot and no alert to screen on"
    return True, ""
