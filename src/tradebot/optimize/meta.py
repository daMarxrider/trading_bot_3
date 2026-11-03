# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Meta-search (#90): tune the optimizer's own settings, ranked by walk-forward, confirmed once on a holdout.

One optimizer run uses fixed settings (random windows on/off, their length, min-symbols-beating,
population, generations). The meta-search tries combinations of them (random search; the user's own
settings always first), runs a walk-forward (#89) with each and ranks them by the mean out-of-sample
score over the folds. The objective and the aggregate stay the user's choice: they define what "better"
means.

Every extra tuned setting is another chance to fit noise, so there are three levels of data::

    |-------- walk-forward folds (rank the combinations) --------|-- holdout --|

The inner optimizer only sees each fold's training part, the combinations are ranked on the folds' test
parts, and the holdout (the last candles) is seen by nothing until the winner is tuned on the stretch
before it and scored there once. Only that number is the result. The report also says how many
combinations were tried and how far the winner is ahead of the median one: after enough tries some
combination always looks good by luck.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import time
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from tradebot.algorithms.base import BaseAlgorithm
from tradebot.optimize.base import Progress, select_frames
from tradebot.optimize.walkforward import WalkForwardResult, plan, walk_forward

#: The optimizer's defaults for the tuned settings (used when the caller doesn't set them).
DEFAULTS = {"samples": 0, "sample_length": 250, "min_symbols_beating": 0, "population": 256, "generations": 30}


@dataclass
class Trial:
    """One tried combination of settings and its walk-forward."""

    index: int
    settings: dict
    result: WalkForwardResult | None
    seconds: float
    error: str | None = None  #: why the combination couldn't run (e.g. windows longer than a fold)

    @property
    def score(self) -> float | None:
        """Mean out-of-sample objective over the folds (the ranking)."""
        return self.result.mean if self.result is not None else None


@dataclass
class MetaResult:
    trials: list[Trial]  #: in the order they were tried; the first has the caller's settings
    final: WalkForwardResult | None  #: the winner's folds plus its holdout, scored once
    space: dict[str, list]
    combinations: int  #: how many distinct combinations the space has
    holdout_length: int
    objective: str = "sharpe"
    stopped: str = "done"  #: "done" (budget or all combinations), "time" or "stopped"

    def ranked(self) -> list[Trial]:
        """The combinations with a score, best first."""
        return sorted((t for t in self.trials if t.score is not None), key=lambda t: -t.score)

    @property
    def best(self) -> Trial | None:
        return next(iter(self.ranked()), None)

    @property
    def median(self) -> float | None:
        scores = [t.score for t in self.ranked()]
        return float(np.median(scores)) if scores else None

    @property
    def lead(self) -> float | None:
        """How far the winner is ahead of the median combination."""
        return self.best.score - self.median if self.best is not None else None

    @property
    def noise(self) -> float | None:
        """The standard error of the winner's score over its folds: a lead below it may be luck."""
        if self.best is None or len(s := self.best.result.scores) < 2:
            return None
        return float(s.std() / math.sqrt(len(s)))

    @property
    def holdout_score(self) -> float | None:
        return self.final.holdout.out_of_sample if self.final is not None else None

    def warnings(self) -> list[str]:
        """Plain-language flags about the search itself; :attr:`final` has the winner's own (folds, holdout)."""
        best, out = self.best, []
        if best is None:
            return ["No combination produced an out-of-sample score."]
        tried = sum(t.result is not None for t in self.trials)
        if tried >= 3 and self.noise is not None and self.lead <= self.noise:
            out.append(f"The best of {tried} combinations is only {self.lead:.3f} ahead of the median one, less than "
                       f"its own fold-to-fold noise ({self.noise:.3f}): the settings hardly matter, or it got lucky.")
        if tried >= 3 and (h := self.holdout_score) is not None and h < self.median:
            out.append(f"On the holdout the winner scores {h:.3f}, below the median combination's walk-forward "
                       f"score ({self.median:.3f}): the meta-search probably fit the folds, not the market.")
        return out  # whether it beats buy & hold on the holdout: final.warnings()


def tuned_space(n_symbols: int = 1, train_length: int | None = None, warmup: int = 0,
                compute: bool = False) -> dict[str, list]:
    """The default choices per tuned setting. Window lengths that don't fit into a fold's training part
    (plus ``warmup`` candles) are left out; population and generations only with ``compute``."""
    space: dict[str, list] = {"samples": [0, 4, 8, 16, 32]}
    lengths = [n for n in (60, 120, 250, 500, 1000) if train_length is None or warmup + n <= train_length]
    if lengths:
        space["sample_length"] = lengths
    else:
        space["samples"] = [0]
    if n_symbols > 1:
        space["min_symbols_beating"] = sorted({0, *(math.ceil(n_symbols * f) for f in (0.25, 0.5, 0.75, 1.0))})
    if compute:
        space["population"] = [64, 128, 256, 512]
        space["generations"] = [15, 30, 60]
    return space


def combinations(space: Mapping[str, list], base: Mapping) -> list[dict]:
    """Every distinct combination; ``sample_length`` only counts with random windows (else ``base``'s)."""
    out = {}
    for values in itertools.product(*space.values()):
        c = {**{k: base[k] for k in DEFAULTS if k in base}, **dict(zip(space, values))}
        if not c.get("samples"):
            c["sample_length"] = base["sample_length"]
        out.setdefault(tuple(sorted(c.items())), c)
    return list(out.values())


def auto_holdout(length: int, folds: int, train_length: int | None = None, test_length: int | None = None,
                 train_fraction: float = 0.7) -> int:
    """A holdout as long as one fold's test stretch."""
    if test_length is not None:
        return test_length
    if train_length is not None:
        return (length - train_length) // (folds + 1)
    ratio = train_fraction / (1 - train_fraction) if train_fraction < 1 else 1.0
    return int(length / (folds + ratio + 1))


def estimate(trials: int, folds: int, space: Mapping[str, list], base: Mapping, method: str = "ga") -> dict:
    """Optimizer runs and fitness evaluations (population × generations, roughly) a meta-search needs."""
    pop = np.mean(space.get("population", [base["population"]]))
    gens = np.mean(space.get("generations", [base["generations"]]))
    runs = trials * folds * (2 if method == "both" else 1) + (2 if method == "both" else 1)
    return {"runs": runs, "evaluations": int(runs * pop * gens)}


def meta_search(
    algorithm: BaseAlgorithm,
    df: pd.DataFrame | Mapping[str, pd.DataFrame],
    trials: int | None = 20,
    folds: int = 5,
    *,
    space: Mapping[str, list] | None = None,
    compute: bool = False,
    time_limit: float | None = None,
    holdout: int | None = None,
    train_length: int | None = None,
    test_length: int | None = None,
    anchored: bool = False,
    method: str = "ga",
    seed: int | None = None,
    progress: Progress | None = None,
    on_phase: Callable[[str], None] | None = None,
    on_start: Callable[[dict, int], None] | None = None,
    on_trial: Callable[[Trial, int], None] | None = None,
    stop: Callable[[], bool] | None = None,
    **kwargs,
) -> MetaResult:
    """Try up to ``trials`` combinations of the optimizer's settings (None: all of them), each with a
    walk-forward of ``folds`` folds, then score the winner once on the holdout.

    ``space`` maps settings to their choices (default :func:`tuned_space`, with ``compute`` also population
    and generations); settings not in it keep the value from ``kwargs``, which go to the optimizer like in
    :func:`walk_forward`. ``holdout`` is in candles (None: one fold's test length; it can't be 0).
    The search ends early after ``time_limit`` seconds (the running combination finishes), when ``stop()``
    returns True or on Ctrl+C; the holdout is scored either way. ``on_start(space, total)`` comes before
    the first combination (e.g. for :func:`estimate`), ``on_trial(trial, total)`` after every one, ``on_phase`` gets labels like "combination 3/20 · fold 2/5"."""
    base = {k: kwargs.pop(k, v) for k, v in DEFAULTS.items()}
    multi = not isinstance(df, pd.DataFrame)
    frames, _ = select_frames(df if multi else {"": df}, kwargs.get("align", False), kwargs.get("min_history", 0.5))
    length = len(reduce(lambda a, b: a.union(b), (f.index for f in frames.values())))
    fraction = kwargs.get("train_fraction", 0.7)
    if holdout is None:
        holdout = max(auto_holdout(length, folds, train_length, test_length, fraction), 3)
    if holdout < 3:
        raise ValueError("the meta-search needs a holdout of at least 3 candles: it is the only honest score")
    windows = plan(length, folds, train_length, test_length, anchored, holdout, fraction)
    if space is None:
        warmup = int(max((s.high for s in algorithm.param_space().values() if s.integer), default=0))
        space = tuned_space(len(frames), windows[0][1] - windows[0][0], warmup, compute)
    if unknown := set(space) - set(DEFAULTS):
        raise ValueError(f"can't tune {', '.join(sorted(unknown))}; choose from {', '.join(DEFAULTS)}")
    combos = combinations(space, base)
    rng = np.random.default_rng(seed)
    first = combinations({}, base)[0]
    order = [first] + [combos[i] for i in rng.permutation(len(combos)) if combos[i] != first]
    total = len(order) if trials is None else min(trials, len(order))
    wf = dict(train_length=train_length, test_length=test_length, anchored=anchored, holdout=holdout, method=method,
              progress=progress, **kwargs)
    if on_start:
        on_start(dict(space), total)
    started, stopped, done = time.monotonic(), "done", []
    for i, settings in enumerate(order[:total]):
        if stop and stop():
            stopped = "stopped"
            break
        if time_limit is not None and time.monotonic() - started >= time_limit:
            stopped = "time"
            break
        phase = (lambda name, i=i: on_phase(f"combination {i + 1}/{total} · {name}")) if on_phase else None
        t0 = time.monotonic()
        try:
            res = walk_forward(algorithm, df, folds, on_phase=phase, parts="folds",
                               seed=None if seed is None else seed + i, **settings, **wf)
            trial = Trial(i, settings, res, time.monotonic() - t0)
        except ValueError as e:
            trial = Trial(i, settings, None, time.monotonic() - t0, str(e))
        except KeyboardInterrupt:
            stopped = "stopped"
            break
        done.append(trial)
        if on_trial:
            on_trial(trial, total)
    result = MetaResult(done, None, dict(space), len(combos), holdout, kwargs.get("objective", "sharpe"), stopped)
    if (best := result.best) is not None:
        phase = (lambda name: on_phase(f"winner · {name}")) if on_phase else None
        last = walk_forward(algorithm, df, folds, on_phase=phase, parts="holdout",
                            seed=None if seed is None else seed + best.index, **best.settings, **wf)
        result.final = dataclasses.replace(best.result, holdout=last.holdout)
    return result
