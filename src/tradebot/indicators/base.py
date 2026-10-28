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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd

from tradebot.core.registry import Param, Registry, resolve_params
from tradebot.core.series import OHLC

INDICATORS: Registry["BaseIndicator"] = Registry("indicator")


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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
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

    @property
    def label(self) -> str:
        args = ",".join(str(v) for v in self.values.values())
        return f"{self.name}({args})" if args else self.name
