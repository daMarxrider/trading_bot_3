# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Monte Carlo robustness test of a backtest (#151).

A backtest is one ordering of the trades. Re-drawing the trades (or the bar returns) thousands of times
shows how much of the final return and especially the maximum drawdown is luck in the order:

* ``shuffle``: the same trades in random order. The final return is the same in every run (multiplication
  commutes), only the drawdown varies.
* ``bootstrap``: draw with replacement, so the final return varies too.
* ``block``: bootstrap in runs of ``block_length`` consecutive items, which keeps streaks together.

Everything is vectorised over the runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METHODS = ("shuffle", "bootstrap", "block")
SOURCES = ("trades", "returns")
BAND = (5.0, 50.0, 95.0)  # percentiles of the confidence band


def _block(n: int, block_length: int | None) -> int:
    return max(1, min(block_length or round(n**0.5), n))


def resample(returns: np.ndarray, runs: int, method: str = "bootstrap", block_length: int | None = None,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """``runs`` re-orderings / re-draws of ``returns``, shape ``[runs, len(returns)]``."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n == 0:
        raise ValueError("nothing to resample: the backtest has no trades")
    rng = rng or np.random.default_rng()
    if method == "shuffle":
        return returns[rng.random((runs, n)).argsort(axis=1)]
    if method == "bootstrap":
        return returns[rng.integers(0, n, (runs, n))]
    if method == "block":
        b = _block(n, block_length)
        starts = rng.integers(0, n - b + 1, (runs, -(-n // b)))
        idx = (starts[..., None] + np.arange(b)).reshape(runs, -1)[:, :n]
        return returns[idx]
    raise ValueError(f"unknown method {method!r} ({', '.join(METHODS)})")


def _curves(returns: np.ndarray, capital: float) -> np.ndarray:
    """Equity curves ``[..., n + 1]`` starting at ``capital``."""
    eq = np.cumprod(1 + returns, axis=-1) * capital
    return np.concatenate([np.full(eq.shape[:-1] + (1,), capital), eq], axis=-1)


def _drawdowns(curves: np.ndarray) -> np.ndarray:
    return (1 - curves / np.maximum.accumulate(curves, axis=-1)).max(-1)


@dataclass
class MonteCarloResult:
    method: str
    source: str
    runs: int
    items: int  # trades or bars that were resampled
    capital: float
    block_length: int | None
    dd_threshold: float
    actual_return: float
    actual_drawdown: float
    final_return: np.ndarray  # per run
    max_drawdown: np.ndarray  # per run, positive fractions
    percentiles: tuple = BAND
    curves: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0)))  # [kept runs, items + 1]
    actual_curve: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def band(self, values: np.ndarray) -> list[float]:
        return [float(v) for v in np.percentile(values, self.percentiles)]

    @property
    def return_band(self) -> list[float]:
        return self.band(self.final_return)

    @property
    def drawdown_band(self) -> list[float]:
        return self.band(self.max_drawdown)

    @property
    def prob_loss(self) -> float:
        return float((self.final_return < 0).mean())

    @property
    def prob_drawdown(self) -> float:
        """Share of runs whose maximum drawdown exceeds ``dd_threshold``."""
        return float((self.max_drawdown > self.dd_threshold).mean())

    def fan(self) -> dict[str, np.ndarray]:
        """Equity curves at the band's low / median / high percentile, over the kept runs."""
        lo, mid, hi = np.percentile(self.curves, self.percentiles, axis=0)
        return {"low": lo, "median": mid, "high": hi}

    def rows(self) -> list[tuple[str, float | None, list[float] | None, float | None]]:
        """(label, backtest value, [low, median, high], probability) for tables."""
        return [("final return", self.actual_return, self.return_band, None),
                ("max drawdown", self.actual_drawdown, self.drawdown_band, None),
                ("probability of a loss", None, None, self.prob_loss),
                (f"probability of a drawdown above {self.dd_threshold:.0%}", None, None, self.prob_drawdown)]


def monte_carlo(returns, runs: int = 1000, method: str = "bootstrap", source: str = "trades",
                block_length: int | None = None, dd_threshold: float = 0.2, capital: float = 10_000.0,
                keep: int = 200, seed: int | None = None) -> MonteCarloResult:
    """Monte Carlo test of ``returns``: per-trade (``source="trades"``) or per-bar returns as fractions.

    ``keep`` equity curves are kept for plotting; the statistics use all ``runs``.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    drawn = resample(returns, runs, method, block_length, np.random.default_rng(seed))
    curves = _curves(drawn, capital)
    actual = _curves(returns, capital)
    return MonteCarloResult(
        method=method, source=source, runs=runs, items=len(returns), capital=capital,
        block_length=_block(len(returns), block_length) if method == "block" else None,
        dd_threshold=dd_threshold, actual_return=float(actual[-1] / capital - 1),
        actual_drawdown=float(_drawdowns(actual)), final_return=curves[:, -1] / capital - 1,
        max_drawdown=_drawdowns(curves), curves=curves[: max(keep, 1)], actual_curve=actual)


def from_backtest(res, runs: int = 1000, source: str = "trades", **kwargs) -> MonteCarloResult:
    """Monte Carlo test of a :class:`~tradebot.backtest.BacktestResult`."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r} ({', '.join(SOURCES)})")
    values = res.trades["return"] if source == "trades" else res.returns
    return monte_carlo(values.to_numpy(dtype=float), runs, source=source, **kwargs)
