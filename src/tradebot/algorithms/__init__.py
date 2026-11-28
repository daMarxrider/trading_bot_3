# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
from tradebot.algorithms import combo, elliott, pool  # noqa: F401  (register the Elliott wave strategy, pool, combo)
from tradebot.algorithms.base import ALGORITHMS, BaseAlgorithm
from tradebot.algorithms.builtin import VotingEnsemble, build_algorithm


def _rank(name: str) -> int:
    cls = ALGORITHMS.get(name)
    return 2 if issubclass(cls, combo.Combo) else 1 if issubclass(cls, VotingEnsemble) else 0


def algorithm_names() -> list[str]:
    """Registered algorithm names for menus: single strategies first, then the ensembles (pool, vote), then
    the combos (``combo`` and the saved ones)."""
    return sorted(ALGORITHMS.names(), key=_rank)


def algorithm_group(name: str) -> str:
    """``"strategy"`` for a single strategy, ``"combined"`` for an ensemble or a combo of several (#230)."""
    return "strategy" if _rank(name) == 0 else "combined"


__all__ = ["ALGORITHMS", "BaseAlgorithm", "algorithm_group", "algorithm_names", "build_algorithm"]
