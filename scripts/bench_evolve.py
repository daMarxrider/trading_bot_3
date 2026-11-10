# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Evolve settings benchmark (#101): which settings give the best result on data the search never saw?

    uv run python scripts/bench_evolve.py run    [--methods fast good best auto-good auto-best] [--workers 6]
    uv run python scripts/bench_evolve.py report

Every method runs the same seed on every dataset and is scored on the newest 20 % of the candles (the test
part), which only the final run touches (Auto-configure keeps it out of its own search). Methods:

* ``fast``       the Evolver defaults (population 300, 40 generations, no shuffled-price runs or folds)
* ``good``       ``best_result_settings(level="good")``
* ``best``       ``best_result_settings(level="best")``
* ``auto-good``  ``auto_configure(level="good")``, then a run with the chosen settings
* ``auto-best``  ``auto_configure(level="best")``, then a run with the chosen settings

The datasets are real prices (stocks, crypto, one basket of three stocks, daily and weekly) plus two *random
walks* as a control: there the test score has no business being positive, and the permutation test should
not call the result an edge. Results are appended to ``benchmarks/evolve/results.jsonl`` (finished jobs are
skipped on a restart), ``report`` writes ``benchmarks/evolve/report.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "evolve"
RESULTS = OUT / "results.jsonl"
METHODS = ("fast", "good", "best", "auto-good", "auto-best")
#: name -> (symbols or "random walk", timeframe, candles); a weight for the job order (cost ~ symbols)
DATASETS = {
    "AAPL 1d": (["AAPL"], "1d", 2000),
    "SAP.DE 1d": (["SAP.DE"], "1d", 2000),
    "BTC/USDT 1d": (["BTC/USDT"], "1d", 2000),
    "ETH/USDT 1d": (["ETH/USDT"], "1d", 2000),
    "BLK 1w": (["BLK"], "1w", 1000),
    "basket 1d (MSFT GOOG BLK)": (["MSFT", "GOOG", "BLK"], "1d", 2000),
    "random walk A": ("random walk", "1d", 2000),
    "random walk B": ("random walk", "1d", 2000),
}
SEED = 7  #: the first seed; ``--seeds`` adds more, each a separate run of every method


def random_walk(seed: int, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    open_ = np.r_[100.0, close[:-1]]
    spread = np.abs(rng.normal(0, 0.006, n))
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + spread),
                         "low": np.minimum(open_, close) * (1 - spread), "close": close, "volume": 1000.0},
                        index=pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC"))


def load(name: str):
    symbols, tf, bars = DATASETS[name]
    if symbols == "random walk":
        return random_walk(1 if name.endswith("A") else 2, bars)
    from tradebot.data import MarketDataEngine

    engine = MarketDataEngine()
    frames = {s: engine.get_ohlc(s, tf, None, None, bars=bars) for s in symbols}
    return frames if len(frames) > 1 else next(iter(frames.values()))


def run_job(name: str, method: str, data_file: str, seed: int = SEED) -> dict:
    """One (dataset, method): the whole procedure, scored on the test part."""
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings

    warnings.filterwarnings("ignore")
    from tradebot.evolve import auto_configure, best_result_settings, evolve

    data = pd.read_pickle(data_file)
    frames = [data] if isinstance(data, pd.DataFrame) else list(data.values())
    candles, n = min(len(f) for f in frames), len(frames)
    started = time.monotonic()
    extra = {}
    if method == "fast":
        settings, permutations, folds = {}, 0, 0
    elif method in ("good", "best"):
        settings = best_result_settings(candles, n, level=method)
    else:
        auto = auto_configure(data, level=method.split("-")[1], seed=seed)
        settings = auto.settings
        extra = {"chosen": {k: v for k, v in settings.items() if isinstance(v, (int, float, str, bool))},
                 "candidates": len(auto.trials),
                 "auto_seconds": round(time.monotonic() - started, 1),
                 "best_trial_scores": sorted((round(t.score, 3) for t in auto.trials if t.round == max(
                     x.round for x in auto.trials)), reverse=True)}
    if method != "fast":
        permutations, folds = settings.pop("permutations"), settings.pop("folds")
    res = evolve(data, permutations=permutations, folds=folds, seed=seed, **settings)
    rows = res.rows()
    lead = [r["vs buy & hold"] for r in rows if r["vs buy & hold"] is not None]
    wf = [x for x in (res.walk_forward.scores() if res.walk_forward else []) if x is not None]
    return {
        "dataset": name, "method": method, "seed": seed, "seconds": round(time.monotonic() - started, 1),
        "train": res.aggregated("train"), "validation": res.aggregated("validation"), "test": res.aggregated("test"),
        "lead_vs_buy_hold": float(np.mean(lead)) if lead else None, "trades": res.trades("test"),
        "p": res.permutation.p if res.permutation else None,
        "deflated": res.deflated.probability if res.deflated else None,
        "folds": wf, "rules": str(res.program), "settings": {k: v for k, v in settings.items() if isinstance(v, (int, float, str, bool))},
        **extra,
    }


def done_jobs() -> set[tuple[str, str, int]]:
    if not RESULTS.exists():
        return set()
    return {(r["dataset"], r["method"], r["seed"]) for r in map(json.loads, RESULTS.read_text().splitlines()) if r}


def run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "data"
    cache.mkdir(exist_ok=True)
    names = [d for d in DATASETS if not args.datasets or any(a.lower() in d.lower() for a in args.datasets)]
    files = {}
    for name in names:
        files[name] = str(cache / (name.replace("/", "-").replace(" ", "_") + ".pkl"))
        if not Path(files[name]).exists():  # fixed once loaded, so every method sees the same candles
            pd.to_pickle(load(name), files[name])
    todo = [(d, m, sd) for sd in args.seeds for m in args.methods for d in names if (d, m, sd) not in done_jobs()]
    weight = {"fast": 1, "good": 12, "best": 60, "auto-good": 20, "auto-best": 300}
    todo.sort(key=lambda j: -weight[j[1]] * (3 if "basket" in j[0] else 1))  # longest first, for the workers' balance
    print(f"{len(todo)} jobs, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_job, d, m, files[d], sd): (d, m, sd) for d, m, sd in todo}
        for f in as_completed(futures):
            d, m, sd = futures[f]
            try:
                r = f.result()
            except Exception as e:  # keep going; the job is retried on the next start
                print(f"FAILED {d} / {m} / seed {sd}: {type(e).__name__}: {e}", flush=True)
                continue
            with RESULTS.open("a") as out:
                out.write(json.dumps(r) + "\n")
            print(f"done {d} / {m} / seed {sd}: test {r['test']:.3f} p {r['p']} in {r['seconds']:.0f}s", flush=True)


def report(_args) -> None:
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines() if line]
    df = pd.DataFrame(rows)
    methods = [m for m in METHODS if m in set(df["method"])]
    real = df[~df["dataset"].str.startswith("random")]
    noise = df[df["dataset"].str.startswith("random")]
    f = lambda v, d=3: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"  # noqa: E731
    out = ["# Evolve settings benchmark", "",
           f"Generated by `scripts/bench_evolve.py report` from {len(df)} runs (seeds {sorted(set(df['seed']))}). Score = Sharpe ratio on "
           "the newest 20 % of the candles, which no search saw (Auto-configure keeps it out of its own search).", ""]
    out += ["## Summary", "", "| method | mean test Sharpe (real) | mean gap train-test (real) | mean lead vs buy & hold | "
            "datasets with test > 0 | random walk: mean test | random walk: p <= 0.05 and test > 0 | mean hours |",
            "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        r, z = real[real.method == m], noise[noise.method == m]
        gap = (r["train"] - r["test"]).mean()
        false = int(((z["p"].fillna(1) <= 0.05) & (z["test"] > 0)).sum())
        out.append(f"| {m} | {f(r['test'].mean())} | {f(gap)} | {f(r['lead_vs_buy_hold'].mean(), 2)} | "
                   f"{int((r['test'] > 0).sum())}/{len(r)} | {f(z['test'].mean())} | {false}/{len(z)} | "
                   f"{f(df[df.method == m]['seconds'].mean() / 3600, 2)} |")
    out += ["", "## Per dataset (mean over seeds: test Sharpe · p of the shuffled-price test · minutes)", "",
            "| dataset | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for d in DATASETS:
        sub = df[df.dataset == d]
        if sub.empty:
            continue
        cells = []
        for m in methods:
            x = sub[sub.method == m]
            cells.append("–" if x.empty else f"{f(x['test'].mean(), 2)} · {f(x['p'].mean(), 2)} · "
                                             f"{x['seconds'].mean() / 60:.0f} (n={len(x)})")
        out.append(f"| {d} | " + " | ".join(cells) + " |")
    auto = df[df.method.str.startswith("auto")]
    if len(auto):
        out += ["", "## What Auto-configure chose", "",
                "| dataset | method | depth | size | parsimony | segments | min trades | aggregate | tournament | mutation |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in auto.iterrows():
            c = r["chosen"]
            out.append(f"| {r['dataset']} | {r['method']} | {c['max_depth']} | {c['max_size']} | "
                       f"{c['parsimony']:.4f} | {c['segments']} | {c['min_trades']} | {c['aggregate']} | "
                       f"{c['tournament']} | {c['mutation']:.2f} |")
    (OUT / "report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("--methods", nargs="+", choices=METHODS, default=["fast", "good", "best", "auto-good"])
    r.add_argument("--datasets", nargs="*", help="only datasets whose name contains one of these")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--seeds", nargs="+", type=int, default=[SEED], help="repeat every job with these seeds")
    r.set_defaults(func=run)
    sub.add_parser("report").set_defaults(func=report)
    a = ap.parse_args()
    sys.exit(a.func(a))
