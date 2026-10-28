# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Particle swarm optimization.

Each particle is a parameter set with a velocity. Every iteration it moves by
a blend of its momentum, a pull toward its own best position and a pull toward
the swarm's best position::

    v = w*v + c1*r1*(personal_best - x) + c2*r2*(swarm_best - x)
    x = x + v

Defaults are the standard constriction coefficients (Clerc & Kennedy):
``w = 0.7298``, ``c1 = c2 = 1.49618``. Velocities are clamped to
``max_velocity`` (fraction of each parameter's range), and a particle hitting
a range boundary stops along that axis instead of bouncing out of range.
Integer parameters are rounded when decoded, so nearby particles share
cached evaluations.
"""

from __future__ import annotations

import numpy as np

from tradebot.optimize.base import OptimizationResult, PopulationOptimizer, Progress


class ParticleSwarmOptimizer(PopulationOptimizer):
    method = "pso"

    def __init__(self, algorithm, df, *, inertia: float = 0.7298, cognitive: float = 1.49618,
                 social: float = 1.49618, max_velocity: float = 0.2, **kwargs):
        super().__init__(algorithm, df, **kwargs)
        self.inertia, self.cognitive, self.social, self.max_velocity = inertia, cognitive, social, max_velocity

    def run(self, progress: Progress | None = None) -> OptimizationResult:
        x = self.initial_population()
        v = self.rng.uniform(-self.max_velocity, self.max_velocity, x.shape) / 2
        fit = self.evaluate(x)
        p_best, p_fit = x.copy(), fit.copy()
        g = int(np.argmax(p_fit))
        history = []
        for it in range(self.generations):
            history.append(float(p_fit[g]))
            if progress:
                progress(it, float(p_fit[g]), self.decode(p_best[g]))
            r1, r2 = self.rng.random(x.shape), self.rng.random(x.shape)
            v = self.inertia * v + self.cognitive * r1 * (p_best - x) + self.social * r2 * (p_best[g] - x)
            v = np.clip(v, -self.max_velocity, self.max_velocity)
            x = x + v
            hit = (x < 0.0) | (x > 1.0)
            x = np.clip(x, 0.0, 1.0)
            v[hit] = 0.0  # absorbing walls: stop at the range limit
            fit = self.evaluate(x)
            improved = fit > p_fit
            p_best[improved], p_fit[improved] = x[improved], fit[improved]
            g = int(np.argmax(p_fit))
        history.append(float(p_fit[g]))
        return self.finish(p_best[g], p_fit[g], history)
