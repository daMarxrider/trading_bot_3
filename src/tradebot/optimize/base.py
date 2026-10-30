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

Several symbols at once (#84): pass ``{symbol: frame}`` instead of one frame. One parameter set is then
scored on every symbol (each with its own train/test split and periods per year) and the per-symbol
fitness values are combined by ``aggregate`` (:data:`AGGREGATES`): ``mean``, ``median``, ``min`` (the
worst symbol) or ``mean-std`` (mean minus one standard deviation). Parameters that only work on one
chart score badly on the rest, so the winner is more likely to generalize. Symbols with much less
history than the longest one are skipped (``min_history``), or every symbol is cut to the common date
range first (``align``).

Two optional extras:

* ``min_symbols_beating``: a candidate whose in-sample objective doesn't beat buy & hold's (same
  objective, same candles) on at least that many symbols gets the rejection fitness, like one with
  too few trades.
* ``samples`` (random windows, GPU batching): instead of the whole in-sample part, every generation
  scores the candidates on ``samples`` random windows of ``sample_length`` candles per symbol, each
  with its own random start (so the dates differ between windows and between symbols, and change
  every generation). All windows have the same length, so a vectorized algorithm scores
  ``[windows, candidates, T]`` in one call - one big batch for the GPU instead of one per symbol.
  Each window gets ``warmup`` extra candles before it (the largest integer parameter, usually the
  longest indicator window) so indicators are warmed up; only the window itself is scored. Fitness
  then changes from generation to generation, so the survivors (GA elites, PSO personal bests) are
  scored again on each generation's windows, and the winner is picked from the final population on
  the whole in-sample part. The history is therefore noisy instead of monotonic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Mapping

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

#: How the per-symbol fitness values (rows of ``[symbols, candidates]``) become one fitness per candidate.
AGGREGATES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": lambda f: f.mean(0),
    "median": lambda f: np.median(f, 0),
    "min": lambda f: f.min(0),
    "mean-std": lambda f: f.mean(0) - f.std(0),
}


def objective_value(metrics: dict, objective: str) -> float:
    """The optimizer's objective computed from backtest metrics (for out-of-sample comparison)."""
    if objective == "return":
        return metrics["total_return"]
    if objective == "calmar":
        dd = metrics["max_drawdown"]
        return metrics["cagr"] / dd if dd > 0 else float("inf")
    return metrics["sharpe"]


def carried_by_few(values) -> bool:
    """True when a good average comes from one or two symbols: the median is far below the mean."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return len(v) >= 3 and v.mean() > 0 and np.median(v) < v.mean() / 2


@dataclass
class SymbolResult:
    """The tuned parameters on one symbol of a multi-symbol search (#84)."""

    symbol: str
    train: BacktestResult
    test: BacktestResult | None
    train_baselines: Baselines
    test_baselines: Baselines | None


@dataclass
class _Data:
    """One symbol's split, as the fitness evaluation needs it."""

    symbol: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    ppy: float
    ohlc: OHLC
    ohlc_np: OHLC | None = None
    buy_hold: float = float("nan")  #: buy & hold's in-sample objective (for ``min_symbols_beating``)


@dataclass
class _Windows:
    """One generation's random windows, stacked: ``ohlc`` fields are ``[K, 1, warmup + length]``."""

    ohlc: OHLC
    ohlc_np: list[OHLC]  #: per window, for loop-based algorithms
    ppy: object  #: ``[K, 1]``
    symbol: np.ndarray  #: ``[K]`` index into the optimizer's data
    buy_hold: np.ndarray  #: ``[K]``
    warmup: int


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
    #: multi-symbol search only (#84): every symbol's results (train/test above are the first symbol's)
    symbols: list[SymbolResult] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  #: symbol -> why it was left out
    aggregate: str = "mean"
    objective: str = "sharpe"

    def objectives(self, period: str = "test") -> dict[str, float]:
        """The objective per symbol on ``period`` ("train" or "test"); one entry for a single-symbol search."""
        rows = self.symbols or [SymbolResult("", self.train, self.test, self.train_baselines, self.test_baselines)]
        return {r.symbol: objective_value(getattr(r, period).metrics, self.objective)
                for r in rows if getattr(r, period) is not None}

    def aggregated(self, period: str = "test") -> float | None:
        """The per-symbol objectives on ``period`` combined like the search combined them; None without data."""
        values = np.array(list(self.objectives(period).values()), dtype=float)
        return float(AGGREGATES[self.aggregate](values[:, None])[0]) if len(values) else None

    def beats(self, period: str = "test") -> dict[str, int]:
        """Baseline -> on how many symbols the strategy's XIRR beat it."""
        out: dict[str, int] = {}
        for r in self.symbols:
            if (bl := getattr(r, f"{period}_baselines")) is not None:
                for name in bl.results:
                    out[name] = out.get(name, 0) + (name not in bl.not_beaten)
        return out

    def rows(self) -> list[dict]:
        """One row per symbol for the results tables: objective in/out, out-of-sample XIRR, lead over each
        baseline and trades (None where there is no out-of-sample part)."""
        train, test = self.objectives("train"), self.objectives("test")
        out = []
        for r in self.symbols:
            bl = r.test_baselines
            out.append({"symbol": r.symbol, "in-sample": train[r.symbol], "out-of-sample": test.get(r.symbol),
                        "xirr": bl.strategy_xirr if bl else None,
                        **{f"vs {k}": bl.excess[k] if bl else None for k in r.train_baselines.results},
                        "trades": len(r.test.trades) if r.test is not None else None})
        return out

    @property
    def carried_by_few(self) -> bool:
        """A multi-symbol result whose average comes from one or two symbols (out-of-sample when there is one)."""
        values = self.objectives("test" if self.test is not None else "train").values()
        return bool(self.symbols) and carried_by_few(values)


class PopulationOptimizer(ABC):
    method: ClassVar[str]

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        df: pd.DataFrame | Mapping[str, pd.DataFrame],
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
        aggregate: str = "mean",
        align: bool = False,
        min_history: float = 0.5,
        min_symbols_beating: int = 0,
        samples: int = 0,
        sample_length: int = 250,
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
        if aggregate not in AGGREGATES:
            raise ValueError(f"unknown aggregate {aggregate!r}; choose {', '.join(AGGREGATES)}")
        self.aggregate = aggregate
        self.multi = not isinstance(df, pd.DataFrame)
        frames, self.skipped = select_frames(df if self.multi else {"": df}, align, min_history)
        self.data = []
        for symbol, frame in frames.items():
            split = int(len(frame) * train_fraction) if train_fraction < 1 else len(frame)
            train_df = frame.iloc[:split]
            self.data.append(_Data(symbol, train_df, frame.iloc[split:], infer_periods_per_year(frame.index),
                                   OHLC.from_frame(train_df, device=self.backend.device)))
        if not 0 <= min_symbols_beating <= len(self.data):
            raise ValueError(f"min_symbols_beating must be between 0 and the number of symbols ({len(self.data)})")
        self.min_symbols_beating = min_symbols_beating
        for d in self.data:
            d.buy_hold = float(self._buy_hold(d.ohlc.close, d.ppy, 0))
        if samples < 0 or sample_length < 2:
            raise ValueError("samples must be 0 (off) or more, and sample_length at least 2 candles")
        self.samples, self.sample_length = samples, sample_length
        self.warmup = int(max((s.high for s in algorithm.param_space().values() if s.integer), default=0))
        self._windows: _Windows | None = None
        if samples:
            total = self.warmup + sample_length
            self._sampled = [i for i, d in enumerate(self.data) if len(d.train_df) >= total]
            if not self._sampled:
                raise ValueError(f"windows of {sample_length} candles plus {self.warmup} for warm-up need {total} "
                                 f"in-sample candles; the longest has {max(len(d.train_df) for d in self.data)}")
            for d in self.data:
                if len(d.train_df) < total:
                    log.info("optimize: %s is too short for windows of %d candles, not sampled", d.symbol, sample_length)
        self.evaluations = 0
        self._cache: dict[tuple, float] = {}

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

    # -- random windows -------------------------------------------------------------------------
    def resample(self) -> bool:
        """With ``samples``: draw this generation's windows (and forget the cached fitness). False without."""
        if not self.samples:
            return False
        total = self.warmup + self.sample_length
        rows, symbols = [], []
        for i in self._sampled:
            d = self.data[i]
            arrays = np.stack([d.train_df[c].to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")])
            for start in self.rng.integers(0, len(d.train_df) - total + 1, self.samples):
                rows.append(arrays[:, start:start + total])
                symbols.append(i)
        stack = np.stack(rows)[:, :, None, :]  # [K, 4, 1, total]
        cols = [np.ascontiguousarray(stack[:, j]) for j in range(4)]  # 4 x [K, 1, total]
        device = self.backend.device
        ohlc = OHLC(*[ops.torch.tensor(c, dtype=ops.torch.float64, device=device) for c in cols]) if device else OHLC(*cols)
        ppy = ops.asarray(np.array([[self.data[i].ppy] for i in symbols]), ohlc.close)
        buy_hold = self._buy_hold(ohlc.close, ppy, self.warmup)
        buy_hold = buy_hold.cpu().numpy() if ops.is_torch(buy_hold) else np.asarray(buy_hold)
        self._windows = _Windows(ohlc, [OHLC(*r) for r in rows], ppy, np.array(symbols), buy_hold.reshape(-1),
                                 self.warmup)
        self._cache.clear()
        return True

    def pick(self, genomes: np.ndarray, fitness: np.ndarray) -> tuple[np.ndarray, float]:
        """The best genome; with ``samples`` re-scored on the whole in-sample part first."""
        if self.samples:
            self._windows = None
            self._cache.clear()
            fitness = self.evaluate(genomes)
        best = int(np.argmax(fitness))
        return genomes[best], float(fitness[best])

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

    def _positions(self, d: _Data, group: list[dict]):
        """Target positions ``[len(group), T]`` on one symbol for a group sharing integer params."""
        if not self._vectorized(group):
            # loop-based algorithm: one NumPy evaluation per candidate, then back to the backend
            if d.ohlc_np is None:
                d.ohlc_np = OHLC.from_frame(d.train_df)
            return ops.asarray(self._loop_signals(d.ohlc_np, group), d.ohlc.close)
        return self._signals(d.ohlc, group)

    def _vectorized(self, group: list[dict]) -> bool:
        vectorized_for = getattr(self.algorithm, "vectorized_for", None)
        return vectorized_for(group[0]) if vectorized_for else getattr(self.algorithm, "vectorized", True)

    def _loop_signals(self, o: OHLC, group: list[dict]) -> np.ndarray:
        length = o.close.shape[-1]
        return np.stack([np.nan_to_num(np.broadcast_to(np.asarray(self.algorithm.signals(o, p), float), (length,)))
                         for p in group])

    def _signals(self, o: OHLC, group: list[dict]):
        """One call of the algorithm for the whole group: float params become ``[group, 1]`` arrays."""
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
        """Fitness per candidate: the per-symbol (or per-window) fitness values combined by ``aggregate``,
        minus complexity; rejected if it beats buy & hold on fewer than ``min_symbols_beating`` symbols."""
        if (w := self._windows) is not None:
            if self._vectorized(group):
                pos = self._signals(w.ohlc, group)  # [K, group, T]: all windows in one call
            else:
                pos = ops.asarray(np.stack([self._loop_signals(o, group) for o in w.ohlc_np]), w.ohlc.close)
            fits = self._score(w.ohlc.close, pos, w.ppy, w.warmup, len(group))
            fits = fits.cpu().numpy() if ops.is_torch(fits) else np.asarray(fits)
            symbol, buy_hold = w.symbol, w.buy_hold
        else:
            fits = np.stack([self._symbol_fitness(d, group) for d in self.data])
            symbol, buy_hold = np.arange(len(self.data)), np.array([d.buy_hold for d in self.data])
        fit = AGGREGATES[self.aggregate](fits)
        if self.min_symbols_beating:
            beating = sum(fits[symbol == i].mean(0) > buy_hold[symbol == i].mean() for i in np.unique(symbol))
            fit = np.where(beating < self.min_symbols_beating, -1e9, fit)
        complexity = getattr(self.algorithm, "complexity", None)
        return fit - np.array([complexity(p) for p in group]) if complexity else fit

    def _symbol_fitness(self, d: _Data, group: list[dict]) -> np.ndarray:
        fit = self._score(d.ohlc.close, self._positions(d, group), d.ppy, 0, len(group))
        return fit.cpu().numpy() if ops.is_torch(fit) else np.asarray(fit)

    def _score(self, close, pos, ppy, warmup: int, n: int):
        """Fitness of positions on ``close``: ``[T]`` -> ``[n]``, or stacked windows ``[K, 1, T]`` -> ``[K, n]``.
        The first ``warmup`` candles only warm up the indicators and are not scored."""
        lo = -1.0 if self.algorithm.allow_short else 0.0
        pos = ops.nan_to_num(pos)
        pos = pos.clamp(lo, 1.0) if ops.is_torch(pos) else np.clip(pos, lo, 1.0)
        shape = (*close.shape[:-2], n, close.shape[-1])
        if tuple(pos.shape) != shape:
            pos = pos.expand(shape) if ops.is_torch(pos) else np.broadcast_to(pos, shape)
        close, pos = close[..., warmup:], pos[..., warmup:]
        returns = simulate(close, pos, self.fee)
        fit = batch_fitness(returns, ppy, self.objective)
        changes = ops.abs_(pos - ops.shift(pos, 1, fill=0.0)) > 0
        n_trades = changes.sum(-1) / 2
        return ops.where(n_trades < self.min_trades, -1e9, ops.nan_to_num(fit, -1e9))

    def _buy_hold(self, close, ppy, warmup: int):
        """Buy & hold's objective on ``close`` (``[T]`` -> scalar, ``[K, 1, T]`` -> ``[K, 1]``)."""
        close = close[..., warmup:]
        returns = simulate(close, ops.full(tuple(close.shape), 1.0, close), self.fee)
        fit = ops.nan_to_num(batch_fitness(returns, ppy, self.objective), -1e9)
        return fit.cpu() if ops.is_torch(fit) else fit

    # -- result ---------------------------------------------------------------------------------
    def finish(self, best_genome: np.ndarray, best_fitness: float, history: list[float]) -> OptimizationResult:
        best_params = self.decode(best_genome)
        tuned = self.algorithm.with_params(**best_params)
        per = [self._symbol_result(d, tuned) for d in self.data]
        first = per[0]
        return OptimizationResult(best_params, float(best_fitness), history, self.backend, first.train, first.test,
                                  self.evaluations, self.method, train_baselines=first.train_baselines,
                                  test_baselines=first.test_baselines, symbols=per if self.multi else [],
                                  skipped=self.skipped, aggregate=self.aggregate, objective=self.objective)

    def _symbol_result(self, d: _Data, tuned: BaseAlgorithm) -> SymbolResult:
        train = backtest(d.train_df, tuned, self.fee, periods_per_year=d.ppy)
        test = backtest(d.test_df, tuned, self.fee, periods_per_year=d.ppy) if len(d.test_df) > 2 else None
        kw = dict(fee=self.fee, amount=self.dca_amount, periods_per_year=d.ppy)
        return SymbolResult(d.symbol, train, test, baselines(d.train_df, train, **kw),
                            baselines(d.test_df, test, **kw) if test is not None else None)


def select_frames(frames: Mapping[str, pd.DataFrame], align: bool = False,
                  min_history: float = 0.5) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """The symbols to search on, and the skipped ones with the reason.

    ``align`` cuts every frame to the date range all of them cover. A symbol with fewer than
    ``min_history`` times the candles of the longest one is skipped, so a short history neither
    dominates the aggregate nor shortens everyone else's data.
    """
    frames = {s: f for s, f in frames.items()}
    skipped = {s: "no data" for s, f in frames.items() if len(f) == 0}
    frames = {s: f for s, f in frames.items() if s not in skipped}
    if align and frames:
        start = max(f.index[0] for f in frames.values())
        end = min(f.index[-1] for f in frames.values())
        frames = {s: f.loc[start:end] for s, f in frames.items()}
    longest = max(map(len, frames.values()), default=0)
    for s, f in list(frames.items()):
        if len(f) < max(min_history * longest, 3):
            skipped[s] = f"only {len(f)} of {longest} candles"
            del frames[s]
    if not frames:
        raise ValueError("no symbol has enough data to optimize on"
                         + (f" ({'; '.join(f'{s}: {r}' for s, r in skipped.items())})" if skipped else ""))
    for s, reason in skipped.items():
        log.info("optimize: skipping %s (%s)", s, reason)
    return frames, skipped
