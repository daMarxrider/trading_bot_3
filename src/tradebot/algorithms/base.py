# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Algorithm (strategy) interface.

An algorithm maps OHLC candles to a *target position* per bar: ``1`` = fully
long, ``0`` = flat, ``-1`` = fully short (only when ``allow_short``).
Fractions are allowed. The position decided on bar ``t`` is held from the
close of ``t`` to the close of ``t+1``, identically in backtests and live.

Writing a new algorithm is one small class::

    from tradebot.algorithms.base import BaseAlgorithm
    from tradebot.core import ops
    from tradebot.core.registry import IntParam, Param
    from tradebot.indicators.builtin import RSI

    class RsiDip(BaseAlgorithm):
        name = "rsi_dip"
        params = {"period": IntParam(14, 2, 50), "level": Param(25, 5, 45)}

        def signals(self, o, p):
            rsi = RSI.compute(o, period=p["period"])["value"]
            return ops.latch(rsi < p["level"], rsi > 50)

Event-driven logic that needs a Python loop over bars is fine too: set
``vectorized = False`` and write ``signals`` for NumPy arrays and scalar
parameters (see ``tradebot.algorithms.elliott``).

Rules that keep it GPU-optimizable for free:

* use only :mod:`tradebot.core.ops` / arithmetic on the arrays (no ``np.`` calls
  on them, no Python loops over bars), and
* treat float params as possibly being arrays of shape ``[population, 1]``.
  Integer params are always plain ``int`` (the optimizer groups by them).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd

from tradebot.core.registry import Param, Registry, resolve_params
from tradebot.core.series import OHLC

ALGORITHMS: Registry["BaseAlgorithm"] = Registry("algorithm")


class BaseAlgorithm(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    params: ClassVar[dict[str, Param]] = {}
    allow_short: ClassVar[bool] = False
    #: False for loop-based algorithms that cannot take ``[population, 1]`` parameter arrays;
    #: the optimizer then evaluates candidates one by one on the CPU.
    vectorized: ClassVar[bool] = True
    #: True when ``signals`` also accepts integer params (window lengths) as ``[population, 1]`` arrays,
    #: i.e. only uses ops that take an array period (``sma``, ``ema``, ``window_sum``, ``shift``).
    #: The optimizer then batches candidates with different windows in one call (#9).
    period_batched: ClassVar[bool] = False
    #: False keeps it out of the optimizer's automatic member selection (the ``pool``), e.g. for composites.
    poolable: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ALGORITHMS.register(cls)

    def __init__(self, **params):
        self.values = resolve_params(self.param_space(), params)

    def param_space(self) -> dict[str, Param]:
        """Tunable parameters of *this instance* (composites extend the class-level ``params``)."""
        return dict(self.params)

    @abstractmethod
    def signals(self, o: OHLC, p: dict):
        """Return target positions shaped like ``o.close`` (or broadcast against batched params)."""

    def execute(self, o: OHLC, p: dict):
        """``(positions, fills)``: the target positions and, for strategies that fill inside a candle (a combo
        with ``fills="intrabar"``), :class:`tradebot.backtest.engine.Fills`; None = every fill at the close."""
        return self.signals(o, p), None

    def chart_indicators(self) -> list:
        """Indicator instances the GUI should overlay when this algorithm is shown."""
        return []

    def with_params(self, **params) -> "BaseAlgorithm":
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.values = resolve_params(self.param_space(), {**self.values, **params})
        return clone

    def run(self, df: pd.DataFrame) -> pd.Series:
        """Target positions for a pandas OHLC frame (NaN warm-up -> flat)."""
        return self.run_fills(df)[0]

    def run_fills(self, df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame | None]:
        """``(positions, fills)`` for a pandas OHLC frame: :meth:`run` and the intrabar fills as a frame
        (columns ``adjust`` and ``turnover``, see :class:`tradebot.backtest.engine.Fills`), or None."""
        pos, fills = self.execute(OHLC.from_frame(df), self.values)
        pos = np.nan_to_num(np.broadcast_to(np.asarray(pos, dtype=float), (len(df),)), nan=0.0)
        lo = -1.0 if self.allow_short else 0.0
        return (pd.Series(np.clip(pos, lo, 1.0), index=df.index, name=self.name),
                None if fills is None else fills.frame(df.index))

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.values.items())
        return f"{type(self).__name__}({args})"
