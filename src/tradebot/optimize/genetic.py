# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Genetic-algorithm parameter search with batched (GPU) fitness evaluation.

The population is decoded into parameter sets and grouped by their integer
parameters (window lengths). Within a group the float parameters become
``[group, 1]`` tensors, so one call of the algorithm's ``signals`` scores the
whole group at once - on the GPU when available. Any algorithm written
against :mod:`tradebot.core.ops` gets this for free.

To limit curve fitting the data is split: parameters are evolved on the first
``train_fraction`` of the candles and the winner is reported out-of-sample on
the remainder.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.backtest import BacktestResult, backtest, batch_fitness, simulate
from tradebot.backtest.metrics import infer_periods_per_year
from tradebot.core import ops
from tradebot.core.series import OHLC
from tradebot.optimize.device import Backend, select_backend

log = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    best_params: dict
    best_fitness: float
    history: list[float]
    backend: Backend
    train: BacktestResult
    test: BacktestResult | None
    evaluations: int = 0
    extra: dict = field(default_factory=dict)


class GeneticOptimizer:
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        df: pd.DataFrame,
        *,
        fee: float = 0.001,
        objective: str = "sharpe",
        population: int = 256,
        generations: int = 30,
        elite_fraction: float = 0.1,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.15,
        tournament: int = 3,
        train_fraction: float = 0.7,
        min_trades: int = 2,
        backend: str | Backend = "auto",
        seed: int | None = None,
        fixed: dict | None = None,
    ):
        self.algorithm = algorithm
        self.fee, self.objective = fee, objective
        self.population, self.generations = population, generations
        self.n_elite = max(1, int(population * elite_fraction))
        self.mutation_rate, self.mutation_scale, self.tournament = mutation_rate, mutation_scale, tournament
        self.min_trades = min_trades
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

    # -- encoding -------------------------------------------------------------------------------
    def decode(self, genome: np.ndarray) -> dict:
        out = dict(self.fixed)
        for gene, key in zip(genome, self.keys):
            spec = self.space[key]
            out[key] = spec.clip(spec.low + gene * (spec.high - spec.low))
        return out

    # -- fitness --------------------------------------------------------------------------------
    def evaluate(self, genomes: np.ndarray) -> np.ndarray:
        """Fitness for every genome (row); groups share integer params and run batched."""
        decoded = [self.decode(g) for g in genomes]
        keys = [tuple(p.values()) for p in decoded]
        int_keys = [k for k, s in self.algorithm.param_space().items() if s.integer]
        groups: dict[tuple, list[int]] = {}
        fitness = np.empty(len(genomes))
        for i, params in enumerate(decoded):
            if keys[i] in self._cache:  # integer-heavy searches revisit the same sets a lot
                fitness[i] = self._cache[keys[i]]
            else:
                groups.setdefault(tuple(params[k] for k in int_keys), []).append(i)

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
        self.evaluations += sum(map(len, jobs))
        return fitness

    def _evaluate_group(self, group: list[dict]) -> np.ndarray:
        o = self.ohlc
        params = {}
        for key in self.algorithm.param_space():
            values = [p[key] for p in group]
            if isinstance(values[0], int) or len(set(values)) == 1:
                params[key] = values[0]
            else:
                arr = np.asarray(values, dtype=np.float64)[:, None]
                params[key] = ops.asarray(arr, o.close)
        pos = self.algorithm.signals(o, params)
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
        return fit.cpu().numpy() if ops.is_torch(fit) else np.asarray(fit)

    # -- evolution ------------------------------------------------------------------------------
    def run(self, progress: Callable[[int, float, dict], None] | None = None) -> OptimizationResult:
        dims = len(self.keys)
        pop = self.rng.random((self.population, dims))
        # seed with the algorithm's current parameters so the result is never worse than the default
        pop[0] = [(self.algorithm.values[k] - s.low) / ((s.high - s.low) or 1) for k, s in self.space.items()]
        fit = self.evaluate(pop)
        history = []
        for gen in range(self.generations):
            order = np.argsort(-fit)
            pop, fit = pop[order], fit[order]
            history.append(float(fit[0]))
            if progress:
                progress(gen, float(fit[0]), self.decode(pop[0]))
            children = self._breed(pop, fit, self.population - self.n_elite)
            child_fit = self.evaluate(children)
            pop = np.vstack([pop[: self.n_elite], children])
            fit = np.concatenate([fit[: self.n_elite], child_fit])
        best = int(np.argmax(fit))
        history.append(float(fit[best]))
        best_params = self.decode(pop[best])
        tuned = self.algorithm.with_params(**best_params)
        train = backtest(self.train_df, tuned, self.fee, periods_per_year=self.ppy)
        test = backtest(self.test_df, tuned, self.fee, periods_per_year=self.ppy) if len(self.test_df) > 2 else None
        return OptimizationResult(best_params, float(fit[best]), history, self.backend, train, test, self.evaluations)

    def _breed(self, pop: np.ndarray, fit: np.ndarray, n: int) -> np.ndarray:
        def pick():
            idx = self.rng.integers(0, len(pop), (n, self.tournament))
            return pop[idx[np.arange(n), np.argmax(fit[idx], axis=1)]]

        a, b = pick(), pick()
        alpha = self.rng.uniform(-0.25, 1.25, a.shape)  # BLX-0.25 blend crossover
        child = a + alpha * (b - a)
        mutate = self.rng.random(child.shape) < self.mutation_rate
        child = child + mutate * self.rng.normal(0, self.mutation_scale, child.shape)
        return np.clip(child, 0.0, 1.0)


def optimize(algorithm: BaseAlgorithm, df: pd.DataFrame, **kwargs) -> OptimizationResult:
    progress = kwargs.pop("progress", None)
    return GeneticOptimizer(algorithm, df, **kwargs).run(progress)
