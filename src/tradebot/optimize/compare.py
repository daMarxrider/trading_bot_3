# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Run GA and PSO on the same problem and measure how much they disagree.

Two independent search methods are a cheap robustness check. On the same
data, split, budget and seed:

* **same parameters, similar fitness**: the optimum is well-defined
  (``consistent``)
* **different parameters, similar fitness**: many parameter sets score alike,
  so the "optimum" is mostly noise and unlikely to generalize (``fragile``)
* **different parameters, clearly different fitness**: one method got stuck
  in a weaker local optimum (``divergent``). Prefer the better one, but check
  it out-of-sample.

With member selection (the ``pool``, #21) the ``use.<member>`` genes count
like parameters (0 = same choice, 1 = different), and a member's parameters
are only compared when both methods use it.

Independently of that, if the in-sample winner is the *worse* of the two
out-of-sample, the report flags likely overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.optimize.base import OptimizationResult, Progress
from tradebot.optimize.genetic import GeneticOptimizer
from tradebot.optimize.pso import ParticleSwarmOptimizer

#: Largest per-parameter difference (fraction of the parameter's search range) still counted as "same".
PARAM_TOLERANCE = 0.10
#: Relative in-sample fitness gap below which two results count as "equally good".
FITNESS_TOLERANCE = 0.10

VERDICTS = {
    "consistent": "Both methods converged to the same parameter region with similar scores. "
                  "The optimum looks well-defined.",
    "fragile": "The methods found clearly different parameters that score almost the same. "
               "The optimum is flat or ambiguous, so the exact values are mostly noise and "
               "unlikely to hold up on new data. Prefer robust or default parameters, or test on more data.",
    "divergent": "The methods ended in different regions with clearly different scores; the weaker one "
                 "probably got stuck in a local optimum. Prefer the better result, but check it out-of-sample.",
}


def objective_value(metrics: dict, objective: str) -> float:
    """The optimizer's objective computed from backtest metrics (for out-of-sample comparison)."""
    if objective == "return":
        return metrics["total_return"]
    if objective == "calmar":
        dd = metrics["max_drawdown"]
        return metrics["cagr"] / dd if dd > 0 else float("inf")
    return metrics["sharpe"]


@dataclass
class ComparisonResult:
    ga: OptimizationResult
    pso: OptimizationResult
    objective: str
    #: |GA - PSO| per parameter, as a fraction of that parameter's search range (0 = identical, 1 = opposite ends)
    param_diff: dict[str, float] = field(default_factory=dict)
    fitness_gap: float = 0.0  #: relative in-sample objective difference
    verdict: str = ""
    overfit_warning: bool = False

    @property
    def mean_param_diff(self) -> float:
        return float(np.mean(list(self.param_diff.values()))) if self.param_diff else 0.0

    @property
    def max_param_diff(self) -> float:
        return max(self.param_diff.values(), default=0.0)

    @property
    def agreement(self) -> float:
        """1 = identical parameters, 0 = maximally different (1 - mean normalized distance)."""
        return 1.0 - self.mean_param_diff

    @property
    def members(self) -> dict[str, tuple[bool, bool]]:
        """Pool member -> (used by GA, used by PSO); empty without member selection."""
        return {k[4:]: (bool(self.ga.best_params[k]), bool(self.pso.best_params[k]))
                for k in self.ga.best_params if k.startswith("use.")}

    @property
    def better(self) -> OptimizationResult:
        return self.ga if self.ga.best_fitness >= self.pso.best_fitness else self.pso

    @property
    def oos(self) -> dict[str, float | None]:
        return {r.method: objective_value(r.test.metrics, self.objective) if r.test else None
                for r in (self.ga, self.pso)}

    def explanation(self) -> str:
        text = VERDICTS[self.verdict]
        if self.overfit_warning:
            text += (f" Warning: the in-sample winner ({self.better.method.upper()}) does worse out-of-sample "
                     "than the other result, a typical sign of overfitting.")
        return text


def compare(ga: OptimizationResult, pso: OptimizationResult, space: dict, objective: str = "sharpe") -> ComparisonResult:
    """Quantify the difference between two optimization results over the same parameter ``space``."""
    diffs = {}
    for key, spec in space.items():
        use = f"use.{key.partition('.')[0]}"
        if use in space and use != key and not (ga.best_params[use] and pso.best_params[use]):
            continue  # a member's parameters only matter when both methods use it
        span = (spec.high - spec.low) or 1.0
        diffs[key] = abs(ga.best_params[key] - pso.best_params[key]) / span
    a, b = ga.best_fitness, pso.best_fitness
    gap = abs(a - b) / max(abs(a), abs(b), 1e-9)
    result = ComparisonResult(ga, pso, objective, diffs, gap)
    if result.max_param_diff <= PARAM_TOLERANCE:
        result.verdict = "consistent"
    elif gap <= FITNESS_TOLERANCE:
        result.verdict = "fragile"
    else:
        result.verdict = "divergent"
    oos = result.oos
    if None not in oos.values() and ga.best_fitness != pso.best_fitness:
        winner, loser = (ga, pso) if ga.best_fitness > pso.best_fitness else (pso, ga)
        result.overfit_warning = oos[winner.method] < oos[loser.method]
    return result


def run_comparison(
    algorithm: BaseAlgorithm,
    df: pd.DataFrame,
    progress: Progress | None = None,
    on_phase=None,
    **kwargs,
) -> ComparisonResult:
    """Run GA, then PSO (same data, split, budget and seed) and compare them.

    ``on_phase(method)`` is called before each run (e.g. to reset a progress bar).
    """
    results = {}
    for cls in (GeneticOptimizer, ParticleSwarmOptimizer):
        if on_phase:
            on_phase(cls.method)
        opt = cls(algorithm, df, **kwargs)
        results[cls.method] = opt.run(progress)
    return compare(results["ga"], results["pso"], opt.space, kwargs.get("objective", "sharpe"))
