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
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, ClassVar

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


class PopulationOptimizer(ABC):
    method: ClassVar[str]

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        df: pd.DataFrame,
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
        split = int(len(df) * train_fraction) if train_fraction < 1 else len(df)
        self.train_df, self.test_df = df.iloc[:split], df.iloc[split:]
        self.ppy = infer_periods_per_year(df.index)
        self.ohlc = OHLC.from_frame(self.train_df, device=self.backend.device)
        self.evaluations = 0
        self._cache: dict[tuple, float] = {}
        self._ohlc_np: OHLC | None = None

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

    def _positions(self, group: list[dict]):
        """Target positions ``[len(group), T]`` for a group sharing integer params."""
        o = self.ohlc
        vectorized_for = getattr(self.algorithm, "vectorized_for", None)
        if not (vectorized_for(group[0]) if vectorized_for else getattr(self.algorithm, "vectorized", True)):
            # loop-based algorithm: one NumPy evaluation per candidate, then back to the backend
            if self._ohlc_np is None:
                self._ohlc_np = OHLC.from_frame(self.train_df)
            rows = [np.nan_to_num(np.broadcast_to(np.asarray(self.algorithm.signals(self._ohlc_np, p), float),
                                                  (len(self.train_df),))) for p in group]
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
        o = self.ohlc
        pos = self._positions(group)
        lo = -1.0 if self.algorithm.allow_short else 0.0
        pos = ops.nan_to_num(pos)
        pos = pos.clamp(lo, 1.0) if ops.is_torch(pos) else np.clip(pos, lo, 1.0)
        if pos.ndim == 1 or pos.shape[0] != len(group):
            pos = pos.reshape(1, -1).expand(len(group), -1) if ops.is_torch(pos) else np.broadcast_to(pos, (len(group), pos.shape[-1]))
        returns = simulate(o.close, pos, self.fee)
        fit = batch_fitness(returns, self.ppy, self.objective)
        changes = ops.abs_(pos - ops.shift(pos, 1, fill=0.0)) > 0
        n_trades = changes.sum(-1) / 2
        fit = ops.where(n_trades < self.min_trades, -1e9, ops.nan_to_num(fit, -1e9))
        fit = fit.cpu().numpy() if ops.is_torch(fit) else np.asarray(fit)
        complexity = getattr(self.algorithm, "complexity", None)
        return fit - np.array([complexity(p) for p in group]) if complexity else fit

    # -- result ---------------------------------------------------------------------------------
    def finish(self, best_genome: np.ndarray, best_fitness: float, history: list[float]) -> OptimizationResult:
        best_params = self.decode(best_genome)
        tuned = self.algorithm.with_params(**best_params)
        train = backtest(self.train_df, tuned, self.fee, periods_per_year=self.ppy)
        test = backtest(self.test_df, tuned, self.fee, periods_per_year=self.ppy) if len(self.test_df) > 2 else None
        kw = dict(fee=self.fee, amount=self.dca_amount, periods_per_year=self.ppy)
        return OptimizationResult(best_params, float(best_fitness), history, self.backend, train, test,
                                  self.evaluations, self.method,
                                  train_baselines=baselines(self.train_df, train, **kw),
                                  test_baselines=baselines(self.test_df, test, **kw) if test is not None else None)
