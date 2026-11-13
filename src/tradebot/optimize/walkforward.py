# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Walk-forward validation (#89): tune on one stretch, score on the next unseen one, move forward, repeat.

One train/test split is a single draw: a strategy can look good or bad depending on what the market did
in that one stretch. Walk-forward runs the normal optimizer (GA, PSO or both) once per **fold**::

    |-- train 1 --|- test 1 -|
         |-- train 2 --|- test 2 -|
              |-- train 3 --|- test 3 -|

The test stretches follow each other without gaps and end at the newest candle (or at the holdout).
With ``anchored`` every training window starts at the first candle and grows, otherwise it rolls.
An optional ``holdout`` (the last candles) is touched by no fold: it is scored once at the end, with
parameters tuned on the stretch right before it, and is reported separately.

Lengths are counted in candles on the combined timeline of all symbols (every date any of them has);
each symbol is cut at the same dates. Without lengths, the folds fill the data and train/test keep the
ratio ``train_fraction``.

The result reports every fold (objective in and out of sample, the #19 baselines, XIRR, trades), the
combined out-of-sample score, how much the tuned parameters move between folds, and flags results whose
out-of-sample score swings a lot between folds or that beat buy & hold in only a few of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.backtest import BacktestResult, Baselines, baselines
from tradebot.backtest.baselines import BUY_HOLD, MONTHLY
from tradebot.backtest.engine import from_positions
from tradebot.backtest.metrics import infer_periods_per_year
from tradebot.optimize.base import AGGREGATES, OptimizationResult, Progress, select_frames
from tradebot.optimize.compare import ComparisonResult, run_comparison
from tradebot.optimize.genetic import GeneticOptimizer
from tradebot.optimize.pso import ParticleSwarmOptimizer

METHODS = {"ga": GeneticOptimizer, "pso": ParticleSwarmOptimizer}


@dataclass
class Fold:
    """One walk-forward step: tuned on ``[start, split)``, scored on ``[split, end)`` (``end`` None = last candle)."""

    index: int
    start: pd.Timestamp
    split: pd.Timestamp
    end: pd.Timestamp | None
    result: OptimizationResult
    comparison: ComparisonResult | None = None  #: with ``method="both"``; ``result`` is its in-sample winner
    holdout: bool = False

    @property
    def params(self) -> dict:
        return self.result.best_params

    @property
    def in_sample(self) -> float | None:
        return self.result.aggregated("train")

    @property
    def out_of_sample(self) -> float | None:
        return self.result.aggregated("test")

    def summary(self) -> dict:
        """The row of the per-fold table: objective in/out, out-of-sample XIRR and lead per baseline (mean over
        the symbols), on how many symbols each baseline was beaten, and the out-of-sample trades."""
        r = self.result
        rows = r.symbols or [r]
        bases = [x.test_baselines for x in rows if x.test_baselines is not None]
        mean = lambda values: float(np.mean(values)) if values else None  # noqa: E731
        return {"in-sample": self.in_sample, "out-of-sample": self.out_of_sample,
                "xirr": mean([b.strategy_xirr for b in bases]),
                **{f"vs {k}": mean([b.excess[k] for b in bases]) for k in (BUY_HOLD, MONTHLY)},
                "beats": {k: (sum(k not in b.not_beaten for b in bases), len(bases)) for k in (BUY_HOLD, MONTHLY)},
                "trades": sum(len(x.test.trades) for x in rows if x.test is not None)}

    def beats(self, baseline: str = BUY_HOLD) -> bool:
        """Beat ``baseline`` out of sample (by XIRR) on at least half of the symbols."""
        k, n = self.summary()["beats"][baseline]
        return n > 0 and k >= n / 2


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    holdout: Fold | None
    space: dict  #: the searched parameters (for :meth:`stability`)
    objective: str = "sharpe"
    aggregate: str = "mean"
    method: str = "ga"
    anchored: bool = False
    train_length: int = 0
    test_length: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)  #: the candles ("" = one symbol)
    fee: float = 0.001
    dca_amount: float | None = None

    @property
    def symbols(self) -> list[str]:
        return list(self.frames)

    def stitched(self, symbol: str = "") -> tuple[BacktestResult, Baselines] | None:
        """Every fold's out-of-sample stretch on ``symbol`` joined into one backtest (each stretch traded with its
        own fold's parameters, without the holdout), with the baselines over the same candles; None when no fold
        has an out-of-sample stretch on that symbol."""
        tests = [r.test for f in self.folds for r in (f.result.symbols or [f.result])
                 if getattr(r, "symbol", "") == symbol and r.test is not None]
        if not tests:
            return None
        positions = pd.concat([t.positions for t in tests])
        fills = None
        if any(t.fills is not None for t in tests):  # intrabar fills (#150), each stretch's own
            fills = pd.concat([t.fills if t.fills is not None else
                               pd.DataFrame({"adjust": 0.0, "turnover": 0.0}, index=t.positions.index) for t in tests])
        df = self.frames[symbol].loc[positions.index]
        ppy = infer_periods_per_year(self.frames[symbol].index)
        res = from_positions(df, positions, "walk-forward", {}, self.fee, periods_per_year=ppy, fills=fills)
        return res, baselines(df, res, self.fee, amount=self.dca_amount, periods_per_year=ppy)

    @property
    def scores(self) -> np.ndarray:
        """The out-of-sample objective of every fold that has one."""
        return np.array([s for f in self.folds if (s := f.out_of_sample) is not None], dtype=float)

    @property
    def mean(self) -> float | None:
        return float(self.scores.mean()) if len(self.scores) else None

    @property
    def median(self) -> float | None:
        return float(np.median(self.scores)) if len(self.scores) else None

    @property
    def std(self) -> float | None:
        return float(self.scores.std()) if len(self.scores) else None

    def beat_counts(self) -> dict[str, int]:
        """Baseline -> in how many folds the strategy beat it out of sample."""
        return {k: sum(f.beats(k) for f in self.folds) for k in (BUY_HOLD, MONTHLY)}

    @property
    def unstable(self) -> bool:
        """The out-of-sample score swings more between folds than its average: a lucky fold, not an edge."""
        s = self.scores
        return len(s) >= 2 and s.std() > abs(s.mean())

    @property
    def rarely_beats(self) -> bool:
        """Beat buy & hold in fewer than half of the folds."""
        return bool(self.folds) and self.beat_counts()[BUY_HOLD] < len(self.folds) / 2

    def stability(self) -> dict[str, float]:
        """Parameter -> how far its tuned value moves between folds, as a fraction of its search range
        (0 = the same every time, 1 = from one end to the other). A pool member's parameters only count in
        the folds that use the member; a parameter no fold uses is left out."""
        out = {}
        for key, spec in self.space.items():
            use = f"use.{key.partition('.')[0]}"
            values = [f.params[key] for f in self.folds
                      if not (use in self.space and use != key and not f.params[use])]
            if len(values) >= 2:
                out[key] = (max(values) - min(values)) / ((spec.high - spec.low) or 1.0)
        return out

    def warnings(self) -> list[str]:
        """Plain-language flags for the reports."""
        out = []
        if self.unstable:
            out.append(f"The out-of-sample {self.objective} varies more between folds than its average "
                       f"({self.mean:.3f} ± {self.std:.3f}): the result depends on which stretch you look at.")
        if self.rarely_beats:
            out.append(f"Out-of-sample, the strategy beats buy & hold in only {self.beat_counts()[BUY_HOLD]} of "
                       f"{len(self.folds)} folds.")
        if moving := [k for k, v in self.stability().items() if v > 0.5]:
            out.append(f"These parameters move across more than half their range between folds: "
                       f"{', '.join(moving)}. Their exact tuned values are mostly noise.")
        if self.holdout is not None and not self.holdout.beats():
            out.append("On the holdout, the strategy does not beat buy & hold.")
        return out


def plan(length: int, folds: int, train_length: int | None = None, test_length: int | None = None,
         anchored: bool = False, holdout: int = 0, train_fraction: float = 0.7) -> list[tuple[int, int, int]]:
    """``(start, split, end)`` candle positions of every fold, then of the holdout if there is one."""
    if folds < 1:
        raise ValueError("walk-forward needs at least one fold")
    if not 0 <= holdout < length:
        raise ValueError(f"the holdout ({holdout} candles) must be shorter than the data ({length})")
    usable = length - holdout
    if train_length is None and test_length is None:
        ratio = train_fraction / (1 - train_fraction) if train_fraction < 1 else 1.0
        test_length = int(usable / (folds + ratio))
    if test_length is None:
        test_length = (usable - train_length) // folds
    if train_length is None:
        train_length = usable - folds * test_length
    if train_length < 3 or test_length < 3:
        raise ValueError(f"{folds} folds don't fit into {usable} candles: train {train_length}, test {test_length} "
                         "(at least 3 each); use fewer folds, shorter windows or more data")
    first = usable - train_length - folds * test_length  # leftover candles go before the first fold
    if first < 0:
        raise ValueError(f"{train_length} training + {folds} × {test_length} test candles need "
                         f"{train_length + folds * test_length}, but there are {usable}"
                         + (f" before the holdout of {holdout}" if holdout else ""))
    out = []
    for i in range(folds):
        split = first + train_length + i * test_length
        out.append((first if anchored else split - train_length, split, split + test_length))
    if holdout:
        out.append((first if anchored else usable - train_length, usable, length))
    return out


def walk_forward(
    algorithm: BaseAlgorithm,
    df: pd.DataFrame | Mapping[str, pd.DataFrame],
    folds: int = 5,
    *,
    train_length: int | None = None,
    test_length: int | None = None,
    anchored: bool = False,
    holdout: int = 0,
    method: str = "ga",
    progress: Progress | None = None,
    on_phase: Callable[[str], None] | None = None,
    parts: str = "all",
    **kwargs,
) -> WalkForwardResult:
    """Run the optimizer (``method`` "ga", "pso" or "both") once per fold; ``kwargs`` go to the optimizer.

    ``on_phase(label)`` is called before each optimizer run, e.g. with "fold 2/5", "fold 2/5 PSO" or
    "holdout" (to reset a progress bar). ``parts`` "folds" leaves the holdout out (its candles stay unused),
    "holdout" runs only the holdout (the meta-search #90 scores its winner there once)."""
    if method not in (*METHODS, "both"):
        raise ValueError(f"unknown optimizer {method!r}; choose ga, pso or both")
    if parts not in ("all", "folds", "holdout") or (parts == "holdout" and not holdout):
        raise ValueError(f"parts must be all, folds or holdout (which needs a holdout), not {parts!r}")
    aggregate = kwargs.get("aggregate", "mean")
    if aggregate not in AGGREGATES:
        raise ValueError(f"unknown aggregate {aggregate!r}; choose {', '.join(AGGREGATES)}")
    multi = not isinstance(df, pd.DataFrame)
    frames, skipped = select_frames(df if multi else {"": df}, kwargs.pop("align", False),
                                    kwargs.get("min_history", 0.5))
    timeline = reduce(lambda a, b: a.union(b), (f.index for f in frames.values()))
    windows = plan(len(timeline), folds, train_length, test_length, anchored, holdout,
                   kwargs.pop("train_fraction", 0.7))
    date = lambda i: timeline[i] if i < len(timeline) else None  # noqa: E731
    out = []
    for n, (start, split, end) in enumerate(windows):
        is_holdout = n == folds
        if parts != "all" and is_holdout != (parts == "holdout"):
            continue
        label = "holdout" if is_holdout else f"fold {n + 1}/{folds}"
        lo, at, hi = date(start), date(split), date(end)
        cut = {s: f[f.index.searchsorted(lo):len(f) if hi is None else f.index.searchsorted(hi)]
               for s, f in frames.items()}
        data = cut if multi else cut[""]
        try:
            if method == "both":
                phase = (lambda m, label=label: on_phase(f"{label} {m.upper()}")) if on_phase else None
                cmp = run_comparison(algorithm, data, progress=progress, on_phase=phase, split_at=at, **kwargs)
                fold = Fold(n, lo, at, hi, cmp.better, cmp, is_holdout)
            else:
                if on_phase:
                    on_phase(label)
                opt = METHODS[method](algorithm, data, split_at=at, **kwargs)
                fold = Fold(n, lo, at, hi, opt.run(progress), holdout=is_holdout)
        except ValueError as e:
            raise ValueError(f"{label}: {e}") from None
        out.append(fold)
    space = {k: v for k, v in algorithm.param_space().items() if k not in (kwargs.get("fixed") or {})}
    last = out.pop() if out and out[-1].holdout else None
    first = windows[0]
    return WalkForwardResult(out, last, space, kwargs.get("objective", "sharpe"), aggregate, method, anchored,
                             first[1] - first[0], first[2] - first[1], skipped, frames, kwargs.get("fee", 0.001),
                             kwargs.get("dca_amount"))
