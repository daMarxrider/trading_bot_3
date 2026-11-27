# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Position sizing (#147): how much to put on, as a separate layer on top of any algorithm's signal.

An algorithm says *when* to be in the market (a target position, 1 = fully invested). :class:`Sized` wraps
one and multiplies that target by a size between 0 and 1 from a sizing model:

* ``fixed``: a fixed fraction of the capital (``size.fraction``);
* ``vol``: volatility targeting, the size that makes the position's annualized volatility ``size.target_vol``
  (realized volatility of the last ``size.window`` candles); quiet markets get a bigger position;
* ``risk``: fixed risk per trade: the stop sits ``size.stop_atr`` ATRs below the entry and the size is
  chosen so that being stopped out loses ``size.risk`` of the capital. It is fixed at the entry and kept
  until the position is closed (also in the live bot);
* ``kelly``: fractional Kelly: ``size.kelly_fraction`` times the Kelly fraction ``mean / variance`` of the
  strategy's own returns while in the market over the last ``size.window`` candles (0 for no edge). Until
  10 such candles are seen the fraction itself is the size.

There is no leverage: the size is capped at 1, so sizing only ever reduces the exposure of the signal.
The sizing parameters are ordinary parameters of the wrapper (names starting with ``size.``), so the
backtester, the optimizer, the walk-forward test and the live bot handle them like any other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param, resolve_params
from tradebot.core.series import OHLC

PREFIX = "size."
MIN_KELLY_BARS = 10

MODELS: dict[str, dict[str, Param]] = {
    "fixed": {
        "fraction": Param(0.5, 0.05, 1.0, "Share of the capital put on at each entry.", share=True),
    },
    "vol": {
        "target_vol": Param(0.2, 0.02, 1.0, "Annual volatility the position aims for (0.2 = 20%). The size shrinks "
                                            "when the market gets wilder and grows when it calms down."),
        "window": IntParam(20, 5, 100, "Candles the volatility is measured over."),
    },
    "risk": {
        "risk": Param(0.01, 0.002, 0.05, "Share of the capital lost when the stop is hit (0.01 = 1%).", share=True),
        "stop_atr": Param(2.0, 0.5, 6.0, "How far below the entry the stop sits, in ATRs. A wider stop means a "
                                         "smaller position."),
        "atr_period": IntParam(14, 5, 50, "Candles in the ATR."),
    },
    "kelly": {
        "kelly_fraction": Param(0.5, 0.1, 1.0, "Share of the Kelly fraction to use (0.5 = half Kelly). Full Kelly "
                                               "is very aggressive.", share=True),
        "window": IntParam(100, 20, 500, "Candles the strategy's recent returns are measured over."),
    },
}
DESCRIPTIONS = {
    "fixed": "Fixed fraction of the capital",
    "vol": "Volatility target: a bigger position in quiet markets",
    "risk": "Fixed risk per trade: the distance to the stop decides the size",
    "kelly": "Fractional Kelly from the strategy's recent returns",
}


def periods_per_year(o: OHLC) -> float:
    """Candles per year, from the candle times (252 without them)."""
    if o.time is None:
        return 252.0
    t = o.time.detach().cpu().numpy() if ops.is_torch(o.time) else np.asarray(o.time)
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    return 365.25 * 86400 / step if step > 0 else 252.0


def _at_last_entry(value, pos):
    """``value`` as it was at the last entry (the last bar where ``pos`` went from flat to invested)."""
    T = pos.shape[-1]
    value = value + pos * 0.0  # broadcast to the positions' shape
    entry = (ops.abs_(pos) > 1e-9) & (ops.abs_(ops.shift(pos, 1, fill=0.0)) <= 1e-9)
    if ops.is_torch(pos):
        import torch

        idx = torch.where(entry, torch.arange(T, device=pos.device), 0)
        last = torch.cummax(idx, dim=-1).values
        return torch.gather(value, -1, last)
    idx = np.where(entry, np.arange(T), 0)
    last = np.maximum.accumulate(idx, axis=-1)
    return np.take_along_axis(value, last, -1)


def multiplier(model: str, o: OHLC, pos, p: dict):
    """The size per candle (0..1) for the signal ``pos``, shaped like ``pos``."""
    close = o.close
    if model == "fixed":
        m = p["size.fraction"] + 0.0 * pos
    elif model == "vol":
        ret = ops.nan_to_num(close / ops.shift(close, 1) - 1)
        vol = ops.rolling_std(ret, p["size.window"]) * periods_per_year(o) ** 0.5
        m = p["size.target_vol"] / ops.maximum(vol, 1e-9)
        m = ops.where(ops.isnan(vol), 1.0, m) + 0.0 * pos  # warm-up: no estimate yet
    elif model == "risk":
        from tradebot.indicators.builtin import ATR

        atr = ATR.compute(o, period=p["size.atr_period"])["value"]
        stop = ops.maximum(p["size.stop_atr"] * atr / close, 1e-9)
        m = ops.where(ops.isnan(stop), 1.0, p["size.risk"] / stop)
        m = _at_last_entry(ops.minimum(m, 1.0), pos)
    elif model == "kelly":
        w = p["size.window"]
        ret = ops.nan_to_num(close / ops.shift(close, 1) - 1)
        held = ops.shift(pos, 1, fill=0.0)
        inm = ops.where(ops.abs_(held) > 1e-9, 1.0, 0.0)
        sret = ops.where(held > 0, 1.0, -1.0) * inm * ret
        n = ops.window_sum(inm, w)
        mean = ops.window_sum(sret, w) / ops.maximum(n, 1.0)
        var = ops.maximum(ops.window_sum(sret * sret, w) / ops.maximum(n, 1.0) - mean * mean, 1e-12)
        k = ops.maximum(mean, 0.0) / var
        m = ops.where(ops.isnan(n) | (n < MIN_KELLY_BARS), 1.0, k) * p["size.kelly_fraction"]
    else:
        raise ValueError(f"unknown sizing model {model!r} ({', '.join(MODELS)})")
    return ops.minimum(ops.maximum(ops.nan_to_num(m, 0.0), 0.0), 1.0)


class Sized(BaseAlgorithm):
    """``inner`` with its target positions scaled by a sizing model. Not registered as an algorithm."""

    poolable = False
    period_batched = False  # the sizing windows are plain integers the optimizer groups by

    def __init__(self, inner: BaseAlgorithm, model: str, **params):
        if model not in MODELS:
            raise ValueError(f"unknown sizing model {model!r}; choose {', '.join(MODELS)}")
        self.inner, self.model = inner, model
        self.name = inner.name
        self.description = f"{inner.description} (sized: {DESCRIPTIONS[model]})"
        self.allow_short = inner.allow_short
        self._size_space = {PREFIX + k: v for k, v in MODELS[model].items()}
        self.values = resolve_params(self.param_space(), {**inner.values, **params})

    @property
    def vectorized(self) -> bool:
        return self.inner.vectorized

    def vectorized_for(self, p: dict) -> bool:
        f = getattr(self.inner, "vectorized_for", None)
        return f(self._inner_params(p)) if f else self.inner.vectorized

    def __getattr__(self, name):  # e.g. a combo's ``fills``, a pool's ``selected``
        if name in ("inner", "_size_space") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def param_space(self):
        return {**self.inner.param_space(), **self._size_space}

    def _inner_params(self, p: dict) -> dict:
        return {k: v for k, v in p.items() if k not in self._size_space}

    def signals(self, o: OHLC, p: dict):
        return self.execute(o, p)[0]

    def split(self, o: OHLC, p: dict):
        """``(signal, fills, size)``: the inner target positions, their fills and the size multiplier."""
        pos, fills = self.inner.execute(o, self._inner_params(p))
        pos = ops.nan_to_num(pos)
        return pos, fills, multiplier(self.model, o, pos, p)

    def execute(self, o: OHLC, p: dict):
        pos, fills, size = self.split(o, p)
        if fills is not None:
            fills = type(fills)(fills.adjust * ops.shift(size, 1, fill=0.0), fills.turnover * size)
        return pos * size, fills

    def run_sized(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """``(signal, size)`` for a pandas frame: the inner target positions and the size multiplier per candle
        (what the live bot uses: it follows the signal and puts on ``stake * size``)."""
        pos, _, size = self.split(OHLC.from_frame(df), self.values)
        flat = lambda x: np.broadcast_to(np.asarray(x, dtype=float), (len(df),))  # noqa: E731
        lo = -1.0 if self.allow_short else 0.0
        return pd.Series(np.clip(flat(pos), lo, 1.0), index=df.index), pd.Series(flat(size), index=df.index)

    def chart_indicators(self):
        return self.inner.chart_indicators()

    def reuse_params(self, params: dict) -> dict:
        inner = self._inner_params(params)
        if hasattr(self.inner, "reuse_params"):
            inner = self.inner.reuse_params(inner)
        return {**inner, **{k: v for k, v in params.items() if k in self._size_space}}


def sized(algorithm: BaseAlgorithm, model: str | None, **params) -> BaseAlgorithm:
    """``algorithm`` with sizing ``model`` (None or "none": unchanged; ``params`` may hold ``size.*`` values)."""
    if not model or model == "none":
        if any(k.startswith(PREFIX) for k in params):
            raise ValueError("size.* parameters need a sizing model (--sizing)")
        return algorithm
    return Sized(algorithm, model, **params)
