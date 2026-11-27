# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""How likely a pair's gap closes before the stop (#223): the Feynman–Kac formula on the Ornstein–Uhlenbeck model.

The z-score of a pair's gap is modelled as mean-reverting, ``dz = -κ z dt + √(2κ) dW``: it is pulled back to 0 at
speed ``κ = ln 2 / half-life`` and its long-run spread is 1 (that is what a z-score is). A trade opened at ``z``
wins when ``|z|`` falls to the exit band before it rises to the stop. By the Feynman–Kac formula the probability
``u`` of that happening within ``N`` candles solves the backward equation of the process::

    u_s = -x u_x + u_xx        x = |z| between exit and stop, s = κ·N
    u(exit) = 1, u(stop) = 0   the barriers
    u(x, s=0) = 0              no time left: the trade didn't close in time

Time only enters as ``κ·N`` (a fast pair over few candles is the same bet as a slow pair over many), so the
equation is solved once per ``(exit, stop)`` into a table over ``x`` and ``s``, and every candle is a lookup. The
solver steps fully implicitly: Crank–Nicolson oscillates on this stiff problem (values up to 1.9 next to the exit),
while implicit steps keep every value between 0 and 1 and never let a longer horizon lower the chance. For ``N → ∞`` the answer has a closed form, ``(S(stop) - S(x)) / (S(stop) - S(exit))`` with
``S' = exp(x²/2)``, which the tests check the table against.

The model is only as good as its two assumptions: that the gap keeps reverting (a pair that breaks apart still
looks mean-reverting until the window has caught up, so the stop stays essential) and that ``κ``, estimated
from the last ``window`` candles, holds for the next ``N``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

#: beyond this κ·N the probability no longer depends on the horizon (it is the closed form to 4 decimals)
S_MAX = 20.0
_NX = 201  #: points across |z| from exit to stop
_STEPS = 8000  #: implicit time steps up to S_MAX (first order in time: fine steps keep the error below 0.5 %)
_ROWS = 400  #: time rows kept, dense at short horizons where the chance changes fastest


@lru_cache(maxsize=16)
def table(exit: float, stop: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(s, x, u)``: the probability ``u[i, j]`` of reaching ``|z| = exit`` before ``|z| = stop`` within rescaled
    time ``s[i] = κ·N`` from ``|z| = x[j]``. ``x`` is uniform, ``s`` is denser near 0."""
    if not 0 <= exit < stop:
        raise ValueError("need 0 <= exit < stop")
    x = np.linspace(exit, stop, _NX)
    h, k = x[1] - x[0], S_MAX / _STEPS
    inner = x[1:-1]
    below = 1 / h**2 + inner / (2 * h)  # weight of u[j-1] in -x u_x + u_xx (central differences)
    above = 1 / h**2 - inner / (2 * h)
    a = np.diag(np.full(len(inner), -2 / h**2)) + np.diag(below[1:], -1) + np.diag(above[:-1], 1)
    boundary = np.zeros(len(inner))
    boundary[0] = below[0]  # u(exit) = 1 feeds the first inner point; u(stop) = 0 feeds nothing
    step = np.linalg.inv(np.eye(len(inner)) - k * a)  # (I - k A) u_new = u_old + k b, an M-matrix: 0 <= u <= 1
    keep = np.unique(np.round(_STEPS * np.linspace(0, 1, _ROWS) ** 2).astype(int))
    u = np.zeros((len(keep), len(x)))
    u[:, 0] = 1.0
    cur = np.zeros(len(inner))
    row = 1  # keep[0] == 0: no time, no chance
    for i in range(1, _STEPS + 1):
        cur = step @ (cur + k * boundary)
        if i == keep[row]:
            u[row, 1:-1] = cur
            row += 1
    s = keep * k
    for arr in (s, x, u):
        arr.setflags(write=False)
    return s, x, u


def probability(z, kappa, horizon: float, exit: float, stop: float) -> np.ndarray:
    """The probability that a trade opened at z-score ``z`` reaches ``|z| <= exit`` before ``|z| >= stop`` within
    ``horizon`` candles, at reversion speed ``kappa`` per candle. Arrays broadcast; NaN where ``z`` or ``kappa`` is
    NaN or the gap doesn't revert (``kappa <= 0``). Inside the exit band it is 1, at or beyond the stop 0."""
    if not horizon > 0:
        raise ValueError(f"the horizon must be at least one candle, not {horizon!r}")
    s_grid, x_grid, u = table(float(exit), float(stop))
    z, kappa = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(kappa, dtype=float))
    x = np.abs(z)
    s = np.clip(np.nan_to_num(kappa * horizon), 0.0, S_MAX)
    i = np.clip(np.searchsorted(s_grid, s, side="right") - 1, 0, len(s_grid) - 2)  # time: dense near 0
    di = (s - s_grid[i]) / (s_grid[i + 1] - s_grid[i])
    fj = np.clip((np.nan_to_num(x) - exit) / (x_grid[1] - x_grid[0]), 0, len(x_grid) - 1 - 1e-9)
    j = fj.astype(int)
    dj = fj - j
    out = (u[i, j] * (1 - di) * (1 - dj) + u[i + 1, j] * di * (1 - dj)
           + u[i, j + 1] * (1 - di) * dj + u[i + 1, j + 1] * di * dj)
    out = np.where(x <= exit, 1.0, np.where(x >= stop, 0.0, out))
    bad = np.isnan(z) | np.isnan(kappa) | ~(kappa > 0)
    return np.where(bad, np.nan, out)


def check_filter(min_probability: float, horizon: float) -> tuple[float, float]:
    """The entry filter's settings, validated: ``min_probability`` from 0 (off) to 1, ``horizon`` in candles >= 1."""
    try:
        p, n = float(min_probability), float(horizon)
    except (TypeError, ValueError):
        raise ValueError(f"min_probability and horizon must be numbers, not {min_probability!r} / {horizon!r}") from None
    if not 0 <= p <= 1:
        raise ValueError(f"the minimum probability must be from 0 (off) to 1, not {min_probability!r}")
    if not (n >= 1 and np.isfinite(n)):
        raise ValueError(f"the horizon must be at least one candle, not {horizon!r}")
    return p, n


def reversion_speed(zscore: pd.Series, window: int = 60) -> pd.Series:
    """κ per candle from an AR(1) fit ``z[t] = c + φ·z[t-1]`` over the ``window`` candles before each bar:
    ``κ = -ln φ``, with φ corrected for the small-sample bias of the fit (Kendall, 1954: ``+ (1 + 3φ) / n``). NaN while fewer than ``window`` candles exist or when ``φ`` is not between 0 and 1 (the gap
    doesn't come back). The bar itself never enters its own estimate (no look-ahead)."""
    if window < 3:
        raise ValueError(f"the window needs at least 3 candles, not {window!r}")
    prev = zscore.shift(1)
    roll = dict(window=window, min_periods=window)
    phi = (zscore.rolling(**roll).cov(prev) / prev.rolling(**roll).var()).shift(1)
    phi = phi + (1 + 3 * phi) / window  # Kendall's correction: a short fit makes the gap look faster than it is
    kappa = -np.log(phi.where((phi > 0) & (phi < 1)))
    return kappa.rename("kappa")


def reversion(zscore: pd.Series, horizon: float, exit: float, stop: float, window: int = 60) -> pd.DataFrame:
    """Per candle: ``probability`` (see :func:`probability`) and the ``half_life`` in candles it assumes."""
    kappa = reversion_speed(zscore, window)
    p = probability(zscore.to_numpy(), kappa.to_numpy(), horizon, exit, stop)
    return pd.DataFrame({"probability": p, "half_life": np.log(2) / kappa}, index=zscore.index)
