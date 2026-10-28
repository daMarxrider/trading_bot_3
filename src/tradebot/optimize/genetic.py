# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Genetic algorithm: tournament selection, BLX blend crossover, Gaussian mutation, elitism.

Evaluation, data split and results come from :class:`~tradebot.optimize.base.PopulationOptimizer`.
"""

from __future__ import annotations

import numpy as np

from tradebot.optimize.base import OptimizationResult, PopulationOptimizer, Progress


class GeneticOptimizer(PopulationOptimizer):
    method = "ga"

    def __init__(self, algorithm, df, *, elite_fraction: float = 0.1, mutation_rate: float = 0.2,
                 mutation_scale: float = 0.15, tournament: int = 3, **kwargs):
        super().__init__(algorithm, df, **kwargs)
        self.n_elite = max(1, int(self.population * elite_fraction))
        self.mutation_rate, self.mutation_scale, self.tournament = mutation_rate, mutation_scale, tournament

    def run(self, progress: Progress | None = None) -> OptimizationResult:
        pop = self.initial_population()
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
        return self.finish(pop[best], fit[best], history)

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
