# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The "Best result" preset of the Evolve page and ``tradebot evolve --best-result`` (#101).

The normal defaults are chosen so a run finishes quickly. This preset ignores runtime: it returns the most
thorough settings, scaled to the data at hand, with the checks that tell luck from an edge (shuffled-price runs,
walk-forward folds). Maxing everything out would be wrong (deeper rules and weaker parsimony overfit), so every
value has a reason, listed in :data:`WHY`.

**Measured** (``benchmarks/evolve/README.md``): a bigger search did *not* give a better score on unseen data
than the fast defaults, and the gap between training and test score grew with the effort. What the preset
buys is the *evidence* (p-value, folds), not a better strategy. The values below are reasoned, not tuned.
"""

from __future__ import annotations

#: Why each value is what it is (also shown in the glossary).
WHY = {
    "population / generations": "Search effort, high for a thorough run (higher from scratch, whose space is far "
                                "larger). Measured: it does not improve the unseen score and overfits slightly more.",
    "hall_of_fame": "A long list gives the validation part more candidates to pick from.",
    "max_depth / max_size": "Not maxed: complexity overfits. Scaled to the history, short histories get small rules "
                            "(from scratch the floor is what an RSI needs, depth 8).",
    "parsimony": "Above the default: shorter rules win ties. Auto-configure picked about 0.009 (median) on the "
                 "benchmark, more than these 0.003-0.004; not yet confirmed as better.",
    "aggregate": "mean-std: a rule has to work consistently on every segment and symbol, not on average.",
    "segments / min_trades": "Grow with the number of candles, so each stretch holds enough trades to judge a rule.",
    "permutations": "best: 20 shuffled-price runs, the smallest count that shows p <= 0.05 with room to spare "
                    "(good: 10, which can only show p ~ 0.1).",
    "folds": "Walk-forward folds from the history length; too short a history gets none.",
}


LEVELS = ("good", "best")  #: "good" keeps the run time somewhat reasonable, "best" ignores it


def best_result_settings(candles: int, symbols: int = 1, *, from_scratch: bool = False, level: str = "best") -> dict:
    """Evolver settings (plus ``permutations`` and ``folds`` for :func:`tradebot.evolve.evolve`) for the best
    out-of-sample result, for ``symbols`` symbols with ``candles`` candles each. ``level`` "good" searches
    about half as much and runs fewer shuffled-price runs and folds; "best" ignores the run time. The
    explicit arguments of the Evolver still override these (``{**best_result_settings(...), **explicit}``)."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; choose {', '.join(LEVELS)}")
    good = level == "good"
    n = max(int(candles), 1)
    train = int(n * 0.6)  # validation and test keep 20 % each
    tier = 0 if n < 800 else 1 if n < 3000 else 2
    if from_scratch:
        depth, size = ((10, 70), (12, 100), (14, 130))[tier]
        population, generations = (1000, 120) if good else (2000, 250)
    else:
        depth, size = ((5, 30), (6, 40), (7, 55))[tier]
        population, generations = (500, 60) if good else (1000, 120)
    return dict(
        population=population, generations=generations, max_depth=depth, max_size=size,
        islands=4, tournament=5, crossover=0.6, mutation=0.35, elite=0.02, hall_of_fame=50 if good else 100,
        parsimony=0.003 if good else 0.004, aggregate="mean-std" if symbols > 1 or n >= 800 else "mean", state=True,
        validation=0.2, test=0.2,
        segments=min(6, max(3, train // 300)),
        min_trades=min(20, max(4, train // 100)),
        permutations=10 if good else 20,
        folds=(0 if n < 1500 else 3) if good else (0 if n < 600 else min(5, max(3, n // 1000))),
    )
