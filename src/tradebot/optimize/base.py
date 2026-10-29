# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Shared machinery for population-based parameter search (GA, PSO).

Candidates are vectors in ``[0, 1]^D`` (one dimension per tunable parameter)
that :meth:`PopulationOptimizer.decode` maps onto the parameter ranges.
Fitness evaluation is batched: candidates are grouped by their integer
parameters (window lengths), and within a group the float parameters become
``[group, 1]`` tensors, so one call of the algorithm's ``signals`` scores the
whole group at once - on the GPU when available. Any algorithm written
against :mod:`tradebot.core.ops` gets this for free.

To limit curve fitting the data is split: parameters are searched on the first
``train_fraction`` of the candles and the winner is reported out-of-sample on
the remainder.

An algorithm can take part in the search through three optional hooks (used
by the ``pool``, #21): ``canonical(params, genes)`` maps a decoded candidate
onto its canonical form (equal forms share one cached evaluation),
``complexity(params)`` is subtracted from the fitness, and
``vectorized_for(params)`` tells whether that candidate can be batched.

Several symbols at once (#84): pass ``{symbol: frame}`` instead of one frame. One parameter set is then
scored on every symbol (each with its own train/test split and periods per year) and the per-symbol
fitness values are combined by ``aggregate`` (:data:`AGGREGATES`): ``mean``, ``median``, ``min`` (the
worst symbol) or ``mean-std`` (mean minus one standard deviation). Parameters that only work on one
chart score badly on the rest, so the winner is more likely to generalize. Symbols with much less
history than the longest one are skipped (``min_history``), or every symbol is cut to the common date
range first (``align``).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Mapping

import numpy as np
import pandas as pd

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.backtest import BacktestResult, Baselines, backtest, baselines, batch_fitness, simulate
from tradebot.backtest.metrics import infer_periods_per_year
from tradebot.core import ops
from tradebot.core.series import OHLC
from tradebot.optimize.device import Backend, select_backend

log = logging.getLogger(__name__)

#: ``progress(step, best_fitness, best_params)``
Progress = Callable[[int, float, dict], None]

#: How the per-symbol fitness values (rows of ``[symbols, candidates]``) become one fitness per candidate.
AGGREGATES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": lambda f: f.mean(0),
    "median": lambda f: np.median(f, 0),
    "min": lambda f: f.min(0),
    "mean-std": lambda f: f.mean(0) - f.std(0),
}


def objective_value(metrics: dict, objective: str) -> float:
    """The optimizer's objective computed from backtest metrics (for out-of-sample comparison)."""
    if objective == "return":
        return metrics["total_return"]
    if objective == "calmar":
        dd = metrics["max_drawdown"]
        return metrics["cagr"] / dd if dd > 0 else float("inf")
    return metrics["sharpe"]


def carried_by_few(values) -> bool:
    """True when a good average comes from one or two symbols: the median is far below the mean."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return len(v) >= 3 and v.mean() > 0 and np.median(v) < v.mean() / 2


@dataclass
class SymbolResult:
    """The tuned parameters on one symbol of a multi-symbol search (#84)."""

    symbol: str
    train: BacktestResult
    test: BacktestResult | None
    train_baselines: Baselines
    test_baselines: Baselines | None


@dataclass
class _Data:
    """One symbol's split, as the fitness evaluation needs it."""

    symbol: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    ppy: float
    ohlc: OHLC
    ohlc_np: OHLC | None = None


@dataclass
class OptimizationResult:
    best_params: dict
    best_fitness: float
    history: list[float]
    backend: Backend
    train: BacktestResult
    test: BacktestResult | None
    evaluations: int = 0
    method: str = "ga"
    extra: dict = field(default_factory=dict)
    train_baselines: Baselines | None = None  #: buy & hold and monthly investing on the same split (#19)
    test_baselines: Baselines | None = None
    #: multi-symbol search only (#84): every symbol's results (train/test above are the first symbol's)
    symbols: list[SymbolResult] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  #: symbol -> why it was left out
    aggregate: str = "mean"
    objective: str = "sharpe"

    def objectives(self, period: str = "test") -> dict[str, float]:
        """The objective per symbol on ``period`` ("train" or "test"); one entry for a single-symbol search."""
        rows = self.symbols or [SymbolResult("", self.train, self.test, self.train_baselines, self.test_baselines)]
        return {r.symbol: objective_value(getattr(r, period).metrics, self.objective)
                for r in rows if getattr(r, period) is not None}

    def aggregated(self, period: str = "test") -> float | None:
        """The per-symbol objectives on ``period`` combined like the search combined them; None without data."""
        values = np.array(list(self.objectives(period).values()), dtype=float)
        return float(AGGREGATES[self.aggregate](values[:, None])[0]) if len(values) else None

    def beats(self, period: str = "test") -> dict[str, int]:
        """Baseline -> on how many symbols the strategy's XIRR beat it."""
        out: dict[str, int] = {}
        for r in self.symbols:
            if (bl := getattr(r, f"{period}_baselines")) is not None:
                for name in bl.results:
                    out[name] = out.get(name, 0) + (name not in bl.not_beaten)
        return out

    def rows(self) -> list[dict]:
        """One row per symbol for the results tables: objective in/out, out-of-sample XIRR, lead over each
        baseline and trades (None where there is no out-of-sample part)."""
        train, test = self.objectives("train"), self.objectives("test")
        out = []
        for r in self.symbols:
            bl = r.test_baselines
            out.append({"symbol": r.symbol, "in-sample": train[r.symbol], "out-of-sample": test.get(r.symbol),
                        "xirr": bl.strategy_xirr if bl else None,
                        **{f"vs {k}": bl.excess[k] if bl else None for k in r.train_baselines.results},
                        "trades": len(r.test.trades) if r.test is not None else None})
        return out

    @property
    def carried_by_few(self) -> bool:
        """A multi-symbol result whose average comes from one or two symbols (out-of-sample when there is one)."""
        values = self.objectives("test" if self.test is not None else "train").values()
        return bool(self.symbols) and carried_by_few(values)


class PopulationOptimizer(ABC):
    method: ClassVar[str]

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        df: pd.DataFrame | Mapping[str, pd.DataFrame],
        *,
        fee: float = 0.001,
        objective: str = "sharpe",
        population: int = 256,
        generations: int = 30,
        train_fraction: float = 0.7,
        min_trades: int = 2,
        backend: str | Backend = "auto",
        seed: int | None = None,
        fixed: dict | None = None,
        dca_amount: float | None = None,
        aggregate: str = "mean",
        align: bool = False,
        min_history: float = 0.5,
    ):
        self.algorithm = algorithm
        self.dca_amount = dca_amount  # monthly investing baseline; None = capital / months
        self.fee, self.objective = fee, objective
        self.population, self.generations = population, generations
        self.min_trades = min_trades
        if backend == "auto" and not getattr(algorithm, "vectorized", True):
            backend = "numpy"  # loop-based algorithms run on the CPU anyway; the GPU would only add transfers
        self.backend = backend if isinstance(backend, Backend) else select_backend(backend)
        self.rng = np.random.default_rng(seed)
        self.fixed = fixed or {}
        self.space = {k: v for k, v in algorithm.param_space().items() if k not in self.fixed}
        self.keys = list(self.space)
        if aggregate not in AGGREGATES:
            raise ValueError(f"unknown aggregate {aggregate!r}; choose {', '.join(AGGREGATES)}")
        self.aggregate = aggregate
        self.multi = not isinstance(df, pd.DataFrame)
        frames, self.skipped = select_frames(df if self.multi else {"": df}, align, min_history)
        self.data = []
        for symbol, frame in frames.items():
            split = int(len(frame) * train_fraction) if train_fraction < 1 else len(frame)
            train_df = frame.iloc[:split]
            self.data.append(_Data(symbol, train_df, frame.iloc[split:], infer_periods_per_year(frame.index),
                                   OHLC.from_frame(train_df, device=self.backend.device)))
        self.evaluations = 0
        self._cache: dict[tuple, float] = {}

    @abstractmethod
    def run(self, progress: Progress | None = None) -> OptimizationResult:
        """Search and return the best parameters with in- and out-of-sample backtests."""

    # -- encoding -------------------------------------------------------------------------------
    def decode(self, genome: np.ndarray) -> dict:
        out = dict(self.fixed)
        for gene, key in zip(genome, self.keys):
            spec = self.space[key]
            out[key] = spec.clip(spec.low + gene * (spec.high - spec.low))
        canonical = getattr(self.algorithm, "canonical", None)
        return canonical(out, dict(zip(self.keys, genome))) if canonical else out

    def encode(self, params: dict) -> np.ndarray:
        return np.array([(params[k] - s.low) / ((s.high - s.low) or 1) for k, s in self.space.items()])

    def initial_population(self) -> np.ndarray:
        """Uniform random candidates; the first is the algorithm's current parameters,
        so a search never ends worse than the defaults."""
        pop = self.rng.random((self.population, len(self.keys)))
        pop[0] = np.clip(self.encode(self.algorithm.values), 0.0, 1.0)
        return pop

    # -- fitness --------------------------------------------------------------------------------
    def evaluate(self, genomes: np.ndarray) -> np.ndarray:
        """Fitness for every genome (row); groups share integer params and run batched."""
        decoded = [self.decode(g) for g in genomes]
        keys = [tuple(p.values()) for p in decoded]
        int_keys = [k for k, s in self.algorithm.param_space().items() if s.integer]
        fitness = np.empty(len(genomes))
        pending: dict[tuple, list[int]] = {}  # uncached parameter set -> rows holding it
        for i in range(len(genomes)):
            if keys[i] in self._cache:  # integer-heavy searches revisit the same sets a lot
                fitness[i] = self._cache[keys[i]]
            else:
                pending.setdefault(keys[i], []).append(i)
        groups: dict[tuple, list[int]] = {}
        for rows in pending.values():  # duplicates within the batch are evaluated once
            groups.setdefault(tuple(decoded[rows[0]][k] for k in int_keys), []).append(rows[0])

        def run(indices):
            fitness[indices] = self._evaluate_group([decoded[i] for i in indices])
            for i in indices:
                self._cache[keys[i]] = fitness[i]

        jobs = list(groups.values())
        if self.backend.kind == "numpy":
            with ThreadPoolExecutor() as pool:
                list(pool.map(run, jobs))
        else:
            for job in jobs:
                run(job)
        for rows in pending.values():
            fitness[rows] = fitness[rows[0]]
        self.evaluations += sum(map(len, jobs))
        return fitness

    def _positions(self, d: _Data, group: list[dict]):
        """Target positions ``[len(group), T]`` on one symbol for a group sharing integer params."""
        o = d.ohlc
        vectorized_for = getattr(self.algorithm, "vectorized_for", None)
        if not (vectorized_for(group[0]) if vectorized_for else getattr(self.algorithm, "vectorized", True)):
            # loop-based algorithm: one NumPy evaluation per candidate, then back to the backend
            if d.ohlc_np is None:
                d.ohlc_np = OHLC.from_frame(d.train_df)
            rows = [np.nan_to_num(np.broadcast_to(np.asarray(self.algorithm.signals(d.ohlc_np, p), float),
                                                  (len(d.train_df),))) for p in group]
            return ops.asarray(np.stack(rows), o.close)
        params = {}
        for key in self.algorithm.param_space():
            values = [p[key] for p in group]
            if isinstance(values[0], int) or len(set(values)) == 1:
                params[key] = values[0]
            else:
                arr = np.asarray(values, dtype=np.float64)[:, None]
                params[key] = ops.asarray(arr, o.close)
        return self.algorithm.signals(o, params)

    def _evaluate_group(self, group: list[dict]) -> np.ndarray:
        """Fitness per candidate: the per-symbol fitness values combined by ``aggregate``, minus complexity."""
        fit = AGGREGATES[self.aggregate](np.stack([self._symbol_fitness(d, group) for d in self.data]))
        complexity = getattr(self.algorithm, "complexity", None)
        return fit - np.array([complexity(p) for p in group]) if complexity else fit

    def _symbol_fitness(self, d: _Data, group: list[dict]) -> np.ndarray:
        o = d.ohlc
        pos = self._positions(d, group)
        lo = -1.0 if self.algorithm.allow_short else 0.0
        pos = ops.nan_to_num(pos)
        pos = pos.clamp(lo, 1.0) if ops.is_torch(pos) else np.clip(pos, lo, 1.0)
        if pos.ndim == 1 or pos.shape[0] != len(group):
            pos = pos.reshape(1, -1).expand(len(group), -1) if ops.is_torch(pos) else np.broadcast_to(pos, (len(group), pos.shape[-1]))
        returns = simulate(o.close, pos, self.fee)
        fit = batch_fitness(returns, d.ppy, self.objective)
        changes = ops.abs_(pos - ops.shift(pos, 1, fill=0.0)) > 0
        n_trades = changes.sum(-1) / 2
        fit = ops.where(n_trades < self.min_trades, -1e9, ops.nan_to_num(fit, -1e9))
        return fit.cpu().numpy() if ops.is_torch(fit) else np.asarray(fit)

    # -- result ---------------------------------------------------------------------------------
    def finish(self, best_genome: np.ndarray, best_fitness: float, history: list[float]) -> OptimizationResult:
        best_params = self.decode(best_genome)
        tuned = self.algorithm.with_params(**best_params)
        per = [self._symbol_result(d, tuned) for d in self.data]
        first = per[0]
        return OptimizationResult(best_params, float(best_fitness), history, self.backend, first.train, first.test,
                                  self.evaluations, self.method, train_baselines=first.train_baselines,
                                  test_baselines=first.test_baselines, symbols=per if self.multi else [],
                                  skipped=self.skipped, aggregate=self.aggregate, objective=self.objective)

    def _symbol_result(self, d: _Data, tuned: BaseAlgorithm) -> SymbolResult:
        train = backtest(d.train_df, tuned, self.fee, periods_per_year=d.ppy)
        test = backtest(d.test_df, tuned, self.fee, periods_per_year=d.ppy) if len(d.test_df) > 2 else None
        kw = dict(fee=self.fee, amount=self.dca_amount, periods_per_year=d.ppy)
        return SymbolResult(d.symbol, train, test, baselines(d.train_df, train, **kw),
                            baselines(d.test_df, test, **kw) if test is not None else None)


def select_frames(frames: Mapping[str, pd.DataFrame], align: bool = False,
                  min_history: float = 0.5) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """The symbols to search on, and the skipped ones with the reason.

    ``align`` cuts every frame to the date range all of them cover. A symbol with fewer than
    ``min_history`` times the candles of the longest one is skipped, so a short history neither
    dominates the aggregate nor shortens everyone else's data.
    """
    frames = {s: f for s, f in frames.items()}
    skipped = {s: "no data" for s, f in frames.items() if len(f) == 0}
    frames = {s: f for s, f in frames.items() if s not in skipped}
    if align and frames:
        start = max(f.index[0] for f in frames.values())
        end = min(f.index[-1] for f in frames.values())
        frames = {s: f.loc[start:end] for s, f in frames.items()}
    longest = max(map(len, frames.values()), default=0)
    for s, f in list(frames.items()):
        if len(f) < max(min_history * longest, 3):
            skipped[s] = f"only {len(f)} of {longest} candles"
            del frames[s]
    if not frames:
        raise ValueError("no symbol has enough data to optimize on"
                         + (f" ({'; '.join(f'{s}: {r}' for s, r in skipped.items())})" if skipped else ""))
    for s, reason in skipped.items():
        log.info("optimize: skipping %s (%s)", s, reason)
    return frames, skipped
