# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
from tradebot.optimize.base import OptimizationResult, PopulationOptimizer
from tradebot.optimize.compare import ComparisonResult, compare, run_comparison
from tradebot.optimize.device import Backend, select_backend
from tradebot.optimize.genetic import GeneticOptimizer
from tradebot.optimize.pso import ParticleSwarmOptimizer

OPTIMIZERS: dict[str, type[PopulationOptimizer]] = {"ga": GeneticOptimizer, "pso": ParticleSwarmOptimizer}


def optimize(algorithm, df, method: str = "ga", **kwargs) -> OptimizationResult:
    """Search parameters with ``method`` ("ga" or "pso"); see :func:`run_comparison` for both."""
    progress = kwargs.pop("progress", None)
    try:
        cls = OPTIMIZERS[method]
    except KeyError:
        raise ValueError(f"unknown optimizer {method!r}; choose {', '.join(OPTIMIZERS)}") from None
    return cls(algorithm, df, **kwargs).run(progress)


__all__ = [
    "OPTIMIZERS", "Backend", "ComparisonResult", "GeneticOptimizer", "OptimizationResult",
    "ParticleSwarmOptimizer", "PopulationOptimizer", "compare", "optimize", "run_comparison", "select_backend",
]
