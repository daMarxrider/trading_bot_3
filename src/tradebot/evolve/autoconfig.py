# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Auto-configure (#101): let a search choose the Evolver settings for the data at hand.

An outer successive-halving search over the settings. Every candidate runs a (shortened) evolve on the data
*without its newest* ``test`` *part*; the candidates are ranked by :func:`robustness`, never by the training
score, and the weaker half is dropped before the survivors get a longer run. The test part is not seen by any
of this: it is touched exactly once, by the final run with the chosen settings. Otherwise the overfitting would
only move up one level, into the settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from tradebot.evolve.preset import best_result_settings
from tradebot.evolve.search import Evolver, EvolutionResult, Progress

GAP_WEIGHT = 0.5  #: how much of the training-over-validation gap is subtracted
MIN_TRADES = 3  #: validation trades below this make a candidate worthless
FAILED = -1e9
#: the keys the search varies; the rest comes from the preset
SEARCHED = ("max_depth", "max_size", "parsimony", "segments", "min_trades", "aggregate", "tournament", "mutation")
GUARDS = ("permutations", "folds")


def robustness(result: EvolutionResult) -> float:
    """Out-of-sample quality of one candidate: the validation score minus ``GAP_WEIGHT`` times how far the
    training score lies above it (a big gap means overfitting). Too few validation trades or no validation
    part: :data:`FAILED`."""
    val, train = result.aggregated("validation"), result.aggregated("train")
    if val is None or train is None or result.trades("validation") < MIN_TRADES or not np.isfinite(val):
        return FAILED
    return float(val - GAP_WEIGHT * max(0.0, train - val))


def hold_out(df: pd.DataFrame | Mapping[str, pd.DataFrame], test: float):
    """The data without its newest ``test`` fraction (by date, over all symbols together)."""
    frames = {"": df} if isinstance(df, pd.DataFrame) else dict(df)
    timeline = reduce(lambda a, b: a.union(b), (f.index for f in frames.values()))
    cut = timeline[len(timeline) - int(round(len(timeline) * test))] if test else None
    cropped = {s: (f if cut is None else f[f.index < cut]) for s, f in frames.items()}
    return cropped[""] if isinstance(df, pd.DataFrame) else cropped


@dataclass
class Trial:
    settings: dict
    round: int
    generations: int
    train: float | None
    validation: float | None
    score: float


@dataclass
class AutoConfig:
    settings: dict  #: for :func:`tradebot.evolve.evolve`, final effort, safeguards included
    trials: list[Trial] = field(default_factory=list)  #: every candidate run, all rounds
    test: float = 0.2  #: the part nothing here has seen

    def report(self) -> list[str]:
        """Plain-language lines: the chosen values, and the scores of the last round's candidates."""
        last = max((t.round for t in self.trials), default=0)
        out = ["chosen: " + ", ".join(f"{k} {v:g}" if isinstance(v, (int, float)) else f"{k} {v}"
                                      for k, v in self.settings.items() if k in SEARCHED + GUARDS)]
        for t in sorted((t for t in self.trials if t.round == last), key=lambda t: -t.score):
            fmt = lambda v: "–" if v is None else f"{v:.3f}"  # noqa: E731
            out.append(f"candidate: train {fmt(t.train)} · validation {fmt(t.validation)} · robust {t.score:.3f} · "
                       + ", ".join(f"{k} {t.settings[k]:g}" if isinstance(t.settings[k], (int, float))
                                   else f"{k} {t.settings[k]}" for k in SEARCHED))
        out.append(f"The newest {self.test:.0%} of the data was kept out of this search.")
        return out


def _sample(rng: np.random.Generator, base: dict) -> dict:
    """One candidate around the preset ``base``: complexity limits, parsimony, segments, minimum trades,
    aggregation and selection pressure."""
    out = dict(base)
    out["max_depth"] = int(np.clip(base["max_depth"] + rng.integers(-2, 3), 3, 16))
    out["max_size"] = int(np.clip(base["max_size"] * rng.uniform(0.6, 1.5), 10, 160))
    out["parsimony"] = float(np.exp(rng.uniform(np.log(0.001), np.log(0.02))))
    out["segments"] = int(np.clip(base["segments"] + rng.integers(-1, 2), 2, 8))
    out["min_trades"] = int(np.clip(base["min_trades"] * rng.uniform(0.5, 2.0), 2, 40))
    out["aggregate"] = str(rng.choice(["mean", "min", "mean-std", "median"]))
    out["tournament"] = int(rng.integers(3, 8))
    out["mutation"] = float(rng.uniform(0.2, 0.4))
    out["crossover"] = min(base["crossover"], 1 - out["mutation"])
    return out


def auto_configure(df: pd.DataFrame | Mapping[str, pd.DataFrame], *, from_scratch: bool = False,
                   objective: str = "sharpe", candidates: int = 8, rounds: int = 3, population: int | None = None,
                   generations: int | None = None, seed: int | None = None, progress: Progress | None = None,
                   on_trial: Callable[[int, int, str], None] | None = None, **fixed) -> AutoConfig:
    """Choose the Evolver settings for ``df`` (one frame or ``{symbol: frame}``).

    ``candidates`` settings (the preset among them) run ``rounds`` times, the better half surviving each round
    and getting twice the generations; the last round runs at ``generations`` (default: the preset's, divided
    by 4 per skipped round in the first). ``fixed`` are Evolver settings the search must not change (fee,
    align, state, dca_amount, ...). ``on_trial(done, total, label)`` reports each candidate run; ``progress``
    is the Evolver's own per-generation callback (raise from it to stop). The returned ``settings`` run on
    the *whole* data, test part included, in :func:`tradebot.evolve.evolve`."""
    seed = int(np.random.SeedSequence().generate_state(1)[0]) if seed is None else seed
    rng = np.random.default_rng(seed)
    n_symbols = 1 if isinstance(df, pd.DataFrame) else len(df)
    frames = [df] if isinstance(df, pd.DataFrame) else list(df.values())
    candles = min(len(f) for f in frames)
    base = best_result_settings(candles, n_symbols, from_scratch=from_scratch)
    population = population or base["population"]
    generations = generations or base["generations"]
    test = fixed.pop("test", base["test"])
    search_data = hold_out(df, test)
    pool = [dict(base)] + [_sample(rng, base) for _ in range(max(candidates, 1) - 1)]
    total = sum(max(1, len(pool) >> r) for r in range(rounds))
    trials: list[Trial] = []
    done = 0
    for r in range(rounds):
        gens = max(2, generations // 2 ** (rounds - 1 - r))
        scored = []
        for s in pool:
            kwargs = {k: v for k, v in s.items() if k not in GUARDS}
            kwargs.update(fixed, objective=objective, from_scratch=from_scratch, population=population,
                          generations=gens, test=0.0, validation=0.25, seed=int(rng.integers(1 << 31)))
            if on_trial:
                on_trial(done, total, f"round {r + 1}/{rounds}")
            result = Evolver(search_data, **kwargs).run(progress)
            done += 1
            trials.append(t := Trial(s, r, gens, result.aggregated("train"), result.aggregated("validation"),
                                     robustness(result)))
            scored.append((t.score, len(scored), s))
        scored.sort(key=lambda x: (-x[0], x[1]))
        pool = [s for _, _, s in scored[:max(1, len(scored) // 2)]]
    best = {**pool[0], "population": population, "generations": generations, **fixed}
    best["test"] = test
    return AutoConfig(best, trials, test)
