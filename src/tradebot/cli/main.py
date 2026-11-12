# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""``tradebot`` command line interface."""

from __future__ import annotations

import logging
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.cli import complete
from tradebot.brokers import BROKERS, create_broker
from tradebot.core.registry import load_plugins
from tradebot.indicators import INDICATORS

complete.match_case_insensitively()  # before the commands are built: "sol" TAB -> SOL/USDT
app = typer.Typer(help="Algorithmic trading platform: charts, backtests, optimization, live bots.", no_args_is_help=True,
                  epilog="Not affiliated with or endorsed by any broker. No warranty; use at your own risk. "
                         "Details: tradebot --version")
bot_app = typer.Typer(help="Automated trading bots.", no_args_is_help=True)
app.add_typer(bot_app, name="bot")
cache_app = typer.Typer(help="The disk cache of candles and symbol lists.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")
console = Console()

Params = Annotated[list[str], typer.Option("--param", "-p", help="Algorithm parameter, e.g. -p lower=30",
                                           autocompletion=complete.algorithm_params)]
Members = Annotated[Optional[str], typer.Option(
    help="Comma-separated members for the 'vote' ensemble, or for the 'pool' (strategies and indicators)",
    autocompletion=complete.members)]
TF = Annotated[str, typer.Option("--tf", help="Timeframe: 1m 5m 15m 30m 1h 4h 1d 1w", autocompletion=complete.timeframes)]
Symbol = Annotated[str, typer.Argument(help="e.g. AAPL, SAP.DE, BTC/USDT", autocompletion=complete.symbols)]
Algo = Annotated[str, typer.Argument(help="Algorithm name (tradebot list)", autocompletion=complete.algorithms)]
Broker = Annotated[str, typer.Option(help="Registered broker name", autocompletion=complete.brokers)]
DcaAmount = Annotated[Optional[float], typer.Option(
    help="Amount the monthly-investing baseline invests each month (default: capital / months)")]


def _version(show: bool):
    if show:
        from tradebot import __version__
        from tradebot.legal import DISCLAIMER

        console.print(f"tradebot {__version__}\n\n{DISCLAIMER}", highlight=False)
        raise typer.Exit()


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
          version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True,
                                                help="Show the version and the disclaimer")] = False):
    from tradebot.data.providers import quiet_yfinance

    quiet_yfinance()
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_plugins()


# -- helpers ------------------------------------------------------------------------------------------
def parse_kv(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"expected key=value, got {item!r}")
        out[key.strip()] = float(value)
    return out


def parse_indicator(spec: str):
    """``name`` or ``name:k=v,k=v`` -> indicator instance."""
    name, _, args = spec.partition(":")
    return INDICATORS.get(name)(**parse_kv([a for a in args.split(",") if a]))


def load_data(symbol: str, tf: str, start: str | None, end: str | None, bars: int | None = None) -> pd.DataFrame:
    """``start``/``end`` select a date range; without ``start``, the last ``bars`` candles before ``end``."""
    from tradebot.data import MarketDataEngine

    with console.status(f"fetching {symbol} {tf} ..."):  # a short history is logged as a warning by the engine
        engine = MarketDataEngine()
        if engine.disk is not None:
            engine.disk.maybe_cleanup()  # at most once a day
        df = engine.get_ohlc(symbol, tf, start, end, bars=bars)
    if df.attrs.get("offline"):
        console.print("[yellow]offline: using cached candles, they may be outdated[/]")
    return df


def make_algo(name: str, params: list[str], members: str | None):
    try:
        return build_algorithm(name, parse_kv(params), members.split(",") if members else None)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None


def results_table(title: str, columns: dict[str, dict]) -> Table:
    """One column of metrics per result; a metric a column lacks shows as "–"."""
    from tradebot.backtest.metrics import fmt

    t = Table(title=title)
    t.add_column("metric")
    for name in columns:
        t.add_column(name, justify="right")
    keys = list(dict.fromkeys(k for metrics in columns.values() for k in metrics))
    for k in keys:
        t.add_row(k.replace("_", " "), *(fmt(k, c[k]) if k in c else "–" for c in columns.values()))
    return t


def warn_not_beaten(bl, where: str, who: str = "the strategy") -> None:
    """Say plainly when the strategy didn't beat a baseline (by XIRR) on that part of the data (#19)."""
    if bl is not None and bl.not_beaten:
        console.print(f"[bold red]{where}, {who} does not beat {' or '.join(bl.not_beaten)}[/]")


# -- commands -----------------------------------------------------------------------------------------
@app.command("list")
def list_(kind: Annotated[str, typer.Argument(help="algorithms | indicators | brokers", autocompletion=complete.choices(
        ("algorithms", "strategies"), ("indicators", ""), ("brokers", "")))] = "algorithms"):
    """List registered algorithms, indicators or brokers (built-ins and plugins)."""
    reg = {"algorithms": ALGORITHMS, "indicators": INDICATORS, "brokers": BROKERS}.get(kind)
    if reg is None:
        raise typer.BadParameter("choose algorithms, indicators or brokers")
    t = Table(title=kind)
    t.add_column("name", style="bold cyan")
    t.add_column("details")
    for cls in reg:
        if kind == "brokers":
            detail = f"{cls.execution_mode.value} · {', '.join(a.value for a in cls.asset_classes)}"
        else:
            params = ", ".join(f"{k}={v.default:g}" for k, v in cls.params.items())
            detail = f"{cls.description}  [dim]{params}[/]"
        t.add_row(cls.name, detail)
    console.print(t)


@app.command()
def chart(
    symbol: Symbol,
    tf: TF = "1d",
    bars: Annotated[int, typer.Option(help="Number of candles")] = 150,
    indicator: Annotated[list[str], typer.Option("--indicator", "-i", help="e.g. -i sma:period=50 -i rsi",
                                                 autocompletion=complete.indicator_specs)] = [],
    algo: Annotated[Optional[str], typer.Option(help="Overlay this algorithm's buy/sell signals",
                                                autocompletion=complete.algorithms)] = None,
    param: Params = [],
    members: Members = None,
    line: Annotated[bool, typer.Option(help="Line chart instead of candles")] = False,
):
    """Terminal OHLC chart with indicator overlays and algorithm signals."""
    from tradebot.cli.charts import render_chart

    algorithm = make_algo(algo, param, members) if algo else None
    inds = [parse_indicator(s) for s in indicator]
    if algorithm and not inds:
        inds = algorithm.chart_indicators()
    df = load_data(symbol, tf, None, None, bars + 250)  # extra history warms up indicators
    print(render_chart(df, f"{symbol} {tf}", inds, algorithm, line, window=bars))


@app.command()
def backtest(
    symbol: Annotated[str, typer.Argument(help="e.g. AAPL, SAP.DE, BTC/USDT; several separated by commas",
                                          autocompletion=complete.symbols)],
    algo: Algo,
    tf: TF = "1d",
    start: Annotated[Optional[str], typer.Option(help="ISO date")] = None,
    end: Annotated[Optional[str], typer.Option(help="ISO date")] = None,
    bars: Annotated[int, typer.Option(help="Number of candles (ignored with --start)")] = 1000,
    param: Params = [],
    members: Members = None,
    fee: Annotated[float, typer.Option(help="Fee per side, fraction")] = 0.001,
    capital: float = 10_000.0,
    dca_amount: DcaAmount = None,
    plot: Annotated[bool, typer.Option(help="Show chart with signals and equity")] = True,
):
    """Backtest an algorithm on historical OHLC data, next to buy & hold and monthly investing."""
    from tradebot.backtest import backtest as run_backtest, baselines
    from tradebot.cli.charts import render_chart

    algorithm = make_algo(algo, param, members)
    symbols = _symbols(symbol, [])
    if len(symbols) > 1:
        _backtest_many(symbols, algorithm, tf, start, end, bars, fee, capital, dca_amount)
        return
    df = load_data(symbol, tf, start, end, bars)
    res = run_backtest(df, algorithm, fee=fee, initial_capital=capital)
    if plot:
        print(render_chart(df, f"{symbol} {tf} · {algorithm!r}", algorithm.chart_indicators(), algorithm, equity=res.equity))
    bl = baselines(df, res, fee=fee, initial_capital=capital, amount=dca_amount)
    console.print(f"[bold]{algorithm!r}[/] on {symbol} {tf} ({df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d})")
    console.print(results_table("backtest", bl.columns("strategy", res)))
    console.print(f"final equity [bold]{res.equity.iloc[-1]:,.2f}[/]")
    warn_not_beaten(bl, "Over this period")


def _backtest_many(symbols, algorithm, tf, start, end, bars, fee, capital, dca_amount) -> None:
    """One row per symbol: the strategy's return, Sharpe, XIRR and lead over each baseline."""
    from tradebot.backtest import backtest as run_backtest, baselines
    from tradebot.backtest.metrics import fmt

    t = Table(title=f"{algorithm!r} · {tf}")
    for col in ("symbol", "total return", "sharpe", "XIRR", "vs buy & hold", "vs monthly", "trades"):
        t.add_column(col, justify="left" if col == "symbol" else "right")
    beats = {}
    for s, df in _load_many(symbols, tf, start, end, bars).items():
        res = run_backtest(df, algorithm, fee=fee, initial_capital=capital)
        bl = baselines(df, res, fee=fee, initial_capital=capital, amount=dca_amount)
        lead = [f"[{'green' if v > 0 else 'red'}]{fmt('vs buy & hold', v)}[/]" for v in bl.excess.values()]
        for name in bl.excess:
            beats[name] = beats.get(name, 0) + (name not in bl.not_beaten)
        t.add_row(s, fmt("total_return", res.metrics["total_return"]), fmt("sharpe", res.metrics["sharpe"]),
                  fmt("xirr", bl.strategy_xirr), *lead, str(len(res.trades)))
    console.print(t)
    n = len(t.rows)
    console.print(" · ".join(f"beats {k} on {v}/{n}" for k, v in beats.items()))


@app.command()
def optimize(
    symbol: Annotated[Optional[str], typer.Argument(
        help="e.g. AAPL, or several separated by commas: AAPL,MSFT,BTC/USDT (can be left out with --watchlist)",
        autocompletion=complete.symbols)] = None,
    algo: Annotated[Optional[str], typer.Argument(
        help="Algorithm name (tradebot list); leave it out to let the optimizer pick from all of them (see --pool)",
        autocompletion=complete.algorithms)] = None,
    tf: TF = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    bars: Annotated[int, typer.Option(help="Number of candles (ignored with --start)")] = 2000,
    members: Members = None,
    pool: Annotated[Optional[str], typer.Option(
        help="Comma-separated strategies and indicators the optimizer picks from (default: all)",
        autocompletion=complete.pool_members)] = None,
    min_members: Annotated[int, typer.Option(help="Use at least this many pool members")] = 1,
    max_members: Annotated[Optional[int], typer.Option(help="Use at most this many pool members")] = None,
    fix: Annotated[list[str], typer.Option(help="Keep a parameter fixed, e.g. --fix period=14",
                                           autocompletion=complete.algorithm_params)] = [],
    method: Annotated[str, typer.Option(help="ga | pso | both (run both and compare them)", autocompletion=complete.choices(
        ("ga", "genetic algorithm"), ("pso", "particle swarm"), ("both", "run both and compare")))] = "ga",
    population: Annotated[int, typer.Option(help="Population (GA) / swarm size (PSO)")] = 256,
    generations: Annotated[int, typer.Option(help="Generations (GA) / iterations (PSO)")] = 30,
    objective: Annotated[str, typer.Option(help="sharpe | return | calmar", autocompletion=complete.choices(
        ("sharpe", ""), ("return", ""), ("calmar", "")))] = "sharpe",
    train: Annotated[float, typer.Option(help="Fraction of data used for training")] = 0.7,
    backend: Annotated[str, typer.Option(help="auto | gpu | cuda | rocm | cpu | numpy", autocompletion=complete.choices(
        ("auto", "GPU if available"), ("gpu", "any GPU"), ("cuda", "NVIDIA only"), ("rocm", "AMD only (alias: amd)"), ("cpu", "torch on the CPU"), ("numpy", "")))] = "auto",
    fee: float = 0.001,
    min_trades: Annotated[int, typer.Option(help="Reject parameter sets with fewer round trips")] = 4,
    dca_amount: DcaAmount = None,
    seed: Optional[int] = None,
    watchlist: Annotated[list[str], typer.Option(
        "--watchlist", "-w", help="Also optimize on this watchlist's symbols (repeatable; 'all' = every watchlist)",
        autocompletion=complete.watchlists)] = [],
    aggregate: Annotated[str, typer.Option(
        help="With several symbols: how their scores combine into one (mean | median | min | mean-std)",
        autocompletion=complete.choices(("mean", ""), ("median", "robust to one outlier"),
                                        ("min", "the worst symbol counts"), ("mean-std", "mean minus spread")))] = "mean",
    align: Annotated[bool, typer.Option(help="With several symbols: cut all of them to their common date range")] = False,
    min_symbols_beating: Annotated[int, typer.Option(
        help="Reject parameter sets that beat buy & hold (in-sample, on the objective) on fewer symbols (0 = off)")] = 0,
    samples: Annotated[int, typer.Option(
        help="Score each generation on this many random windows per symbol instead of all in-sample candles; "
             "new windows every generation, batched in one go on the GPU (0 = off)")] = 0,
    sample_length: Annotated[int, typer.Option(help="Candles per random window (with --samples)")] = 250,
    walk_forward: Annotated[int, typer.Option(
        help="Walk-forward validation: re-optimize in this many folds, each scored on the next unseen stretch "
             "(0 = off: one split by --train)")] = 0,
    wf_train: Annotated[Optional[int], typer.Option(
        help="Walk-forward: training candles per fold (default: from --train and the data)")] = None,
    wf_test: Annotated[Optional[int], typer.Option(
        help="Walk-forward: test candles per fold (default: the folds fill the data)")] = None,
    anchored: Annotated[bool, typer.Option(
        help="Walk-forward: every training window starts at the first candle and grows, instead of rolling")] = False,
    holdout: Annotated[int, typer.Option(
        help="Walk-forward: keep the last N candles out of every fold and score them once at the end "
             "(with --meta: 0 = one fold's test length)")] = 0,
    meta: Annotated[int, typer.Option(
        help="Meta-search: try this many combinations of the optimizer's own settings (random windows, their "
             "length, min-symbols-beating), rank them by walk-forward and score the winner once on the holdout "
             "(0 = off; Ctrl+C stops early)")] = 0,
    meta_time: Annotated[Optional[float], typer.Option(
        help="Meta-search: stop starting new combinations after this many minutes")] = None,
    meta_compute: Annotated[bool, typer.Option(
        help="Meta-search: also tune population and generations")] = False,
):
    """Optimize algorithm parameters with a genetic algorithm and/or particle swarm (GPU accelerated
    when available) and report out-of-sample results. ``--method both`` also measures how much the two
    methods disagree - large disagreement means a fragile optimum.

    Without an algorithm, the optimizer also picks *which* strategies vote and which indicators filter
    the entries, from all of them or from ``--pool``.

    With several symbols (comma-separated, or ``--watchlist``) one parameter set is scored on every
    symbol and the scores are combined (``--aggregate``): settings that only work on one chart lose.

    ``--meta N`` tunes the optimizer's own settings: N combinations, each ranked by a walk-forward
    (``--walk-forward`` folds, default 5), the winner scored once on a holdout nothing else has seen."""
    from rich.progress import Progress

    from tradebot.optimize import OPTIMIZERS, run_comparison, select_backend
    from tradebot.optimize.base import AGGREGATES

    if method not in (*OPTIMIZERS, "both"):
        raise typer.BadParameter("method must be ga, pso or both")
    if aggregate not in AGGREGATES:
        raise typer.BadParameter(f"aggregate must be one of {', '.join(AGGREGATES)}", param_hint="--aggregate")
    if watchlist and algo is None and symbol and (symbol == "pool" or symbol in ALGORITHMS.names()):
        symbol, algo = None, symbol  # tradebot optimize -w Crypto sma_cross
    symbols = _symbols(symbol, watchlist)
    if algo in (None, "pool"):
        algorithm = _make_pool(pool or members, min_members, max_members)
        algo = "pool"
    elif pool or min_members != 1 or max_members is not None:
        raise typer.BadParameter("--pool, --min-members and --max-members need the pool: leave out the algorithm")
    else:
        algorithm = make_algo(algo, [], members)
    if len(symbols) == 1:
        df = load_data(symbols[0], tf, start, end, bars)
        size = f"{len(df)} candles"
    else:
        df = _load_many(symbols, tf, start, end, bars)
        size = f"{len(df)} symbols · {sum(map(len, df.values()))} candles"
    if backend == "auto" and not algorithm.vectorized:
        backend = "numpy"  # loop-based algorithms (e.g. elliott_wave) run on the CPU anyway
    kwargs = dict(population=population, generations=generations, objective=objective, train_fraction=train,
                  backend=select_backend(backend), fee=fee, seed=seed, fixed=parse_kv(fix), min_trades=min_trades,
                  dca_amount=dca_amount, aggregate=aggregate, align=align, min_symbols_beating=min_symbols_beating,
                  samples=samples, sample_length=sample_length)
    n_genes = len([k for k in algorithm.param_space() if k not in kwargs["fixed"]])
    console.print(f"backend: [bold]{kwargs['backend'].description}[/] · {size} · {n_genes} parameters"
                  + (f" · {samples} random windows of {sample_length} candles per symbol and generation" if samples else ""))
    if meta:
        return _meta(algorithm, df, symbols, meta, meta_time, meta_compute, walk_forward or 5, wf_train, wf_test,
                     anchored, holdout or None, method, kwargs)
    symbol = symbols[0]
    with Progress(console=console) as progress:
        task = progress.add_task("optimizing", total=generations)
        label = {"name": method.upper()}

        def phase(name):
            label["name"] = name if walk_forward else name.upper()
            progress.reset(task, total=generations)

        def report(step, best, params):
            progress.update(task, advance=1, description=f"{label['name']} step {step + 1} best {objective}={best:.3f}")

        try:
            if walk_forward:
                from tradebot.optimize import walk_forward as run_walk_forward

                wf =run_walk_forward(algorithm, df, walk_forward, train_length=wf_train, test_length=wf_test,
                                      anchored=anchored, holdout=holdout, method=method, progress=report,
                                      on_phase=phase, **kwargs)
            elif method == "both":
                cmp = run_comparison(algorithm, df, progress=report, on_phase=phase, **kwargs)
            else:
                res = OPTIMIZERS[method](algorithm, df, **kwargs).run(report)
        except ValueError as e:  # e.g. no symbol with enough history
            raise typer.BadParameter(str(e)) from None
    if walk_forward:
        _print_walk_forward(wf, algorithm, ",".join(symbols))
        return
    if done := (cmp.ga if method == "both" else res).symbols:
        symbol = ",".join(r.symbol for r in done)  # for the reuse command, without the skipped symbols
    if method == "both":
        _print_comparison(cmp, symbol, algorithm)
        return
    if hasattr(algorithm, "selected"):
        _print_members(algorithm, {method.upper(): res.best_params})
    t = Table(title=f"best parameters ({method.upper()})")
    t.add_column("param")
    t.add_column("value")
    for k, v in _reused(algorithm, res.best_params).items():
        t.add_row(k, f"{v:g}")
    console.print(t)
    if res.symbols:
        _print_symbols({method.upper(): res})
    else:
        console.print(results_table("in-sample", res.train_baselines.columns("strategy", res.train)))
        if res.test is not None:
            console.print(results_table("out-of-sample", res.test_baselines.columns("strategy", res.test)))
        warn_not_beaten(res.test_baselines, "Out-of-sample")
    console.print(f"{res.evaluations} evaluations · reuse with: [cyan]{_reuse_command(symbol, algorithm, res.best_params)}[/]")


def _meta(algorithm, df, symbols, trials, minutes, compute, folds, wf_train, wf_test, anchored, holdout, method,
          kwargs) -> None:
    """``optimize --meta``: the search with a progress bar, then the combinations, the winner and its holdout (#90)."""
    from rich.progress import Progress

    from tradebot.optimize import meta_search
    from tradebot.optimize.meta import DEFAULTS, estimate

    seed = kwargs.pop("seed")
    with Progress(console=console) as progress:
        task = progress.add_task("meta-search", total=None if compute else kwargs["generations"])
        runs = progress.add_task("combinations", total=trials)
        label = {"name": ""}

        def start(space, total):
            cost = estimate(total, folds, space, {**DEFAULTS, **kwargs}, method)
            progress.console.print(f"meta-search: {total} combinations × {folds} folds + the holdout ≈ {cost['runs']} "
                                   f"optimizer runs, {cost['evaluations']:,} evaluations")
            progress.update(runs, total=total)

        def phase(name):
            label["name"] = name
            progress.reset(task)

        def report(step, best, params):
            progress.update(task, advance=1, description=f"{label['name']} step {step + 1} best {best:.3f}")

        try:
            res = meta_search(algorithm, df, trials, folds, compute=compute, time_limit=minutes and minutes * 60,
                              holdout=holdout, train_length=wf_train, test_length=wf_test, anchored=anchored,
                              method=method, seed=seed, progress=report, on_phase=phase, on_start=start,
                              on_trial=lambda trial, total: progress.advance(runs), **kwargs)
        except ValueError as e:
            raise typer.BadParameter(str(e)) from None
    _print_meta(res, algorithm, ",".join(symbols))


def _print_meta(res, algorithm, symbol: str) -> None:
    """The tried combinations ranked by walk-forward, how far the winner leads, then its folds and holdout."""
    obj = res.objective
    tuned = list(res.space)
    t = Table(title=f"meta-search: {len(res.trials)} of {res.combinations} combinations, ranked by walk-forward")
    for col in ("#", *tuned, f"{obj} out (folds)", "beats buy & hold", "time"):
        t.add_column(col, justify="left" if col == "#" else "right")
    best = res.best
    for trial in res.ranked() + [x for x in res.trials if x.score is None]:
        cells = [f"{trial.settings[k]:g}" if k != "sample_length" or trial.settings["samples"] else "–" for k in tuned]
        if trial.result is None:
            score, beats = f"[red]{trial.error}[/]", ""
        else:
            r = trial.result
            score = f"{r.mean:.3f} ± {r.std:.3f}" if r.mean is not None else "–"
            beats = f"{r.beat_counts()['buy & hold']}/{len(r.folds)}"
        mark = " ★" if trial is best else " (yours)" if trial.index == 0 else ""
        t.add_row(f"{trial.index + 1}{mark}", *cells, score, beats, f"{trial.seconds:.0f} s",
                  style="bold" if trial is best else None)
    console.print(t)
    if res.stopped != "done":
        console.print(f"[yellow]stopped early ({'time limit' if res.stopped == 'time' else 'interrupted'})[/]")
    if best is None:
        for w in res.warnings():
            console.print(f"[bold red]{w}[/]")
        return
    console.print(f"winner #{best.index + 1}: {best.score:.3f} out of sample over the folds, {res.lead:+.3f} ahead of "
                  f"the median combination ({res.median:.3f})"
                  + (f", fold-to-fold noise ±{res.noise:.3f}" if res.noise is not None else ""))
    flags = " ".join(f"--{k.replace('_', '-')} {best.settings[k]:g}" for k in tuned
                     if k != "sample_length" or best.settings["samples"])
    console.print(f"winning settings: [cyan]{flags}[/]")
    _print_walk_forward(res.final, algorithm, symbol)
    h = res.holdout_score
    console.print(f"[bold]result: {obj} {'–' if h is None else f'{h:.3f}'} on the holdout of {res.holdout_length} "
                  "candles, scored once[/] (the walk-forward scores above picked the winner, so they flatter it)")
    for w in res.warnings():
        console.print(f"[bold red]{w}[/]")


def _symbols(symbol: str | None, watchlists: list[str]) -> list[str]:
    """The comma-separated symbols, then those of each ``--watchlist`` ("all": every list), de-duplicated."""
    out = [s.strip() for s in (symbol or "").split(",") if s.strip()]
    for name in watchlists:
        try:
            from tradebot.config import watchlist as members, watchlists as lists

            out += members(None if name == "all" and "all" not in lists() else name)
        except KeyError as e:
            raise typer.BadParameter(e.args[0], param_hint="--watchlist") from None
    if not out:
        raise typer.BadParameter("give a symbol (or several, comma-separated) or --watchlist")
    return list(dict.fromkeys(out))


def _load_many(symbols: list[str], tf: str, start, end, bars) -> dict[str, pd.DataFrame]:
    """Candles per symbol; a symbol that can't be loaded is left out with a note."""
    frames = {}
    for s in symbols:
        try:
            frames[s] = load_data(s, tf, start, end, bars)
        except Exception as e:  # unknown symbol, delisted, offline without cache
            console.print(f"[yellow]skipping {s}: {e}[/]")
    if not frames:
        raise typer.BadParameter("none of the symbols could be loaded")
    return frames


def _print_symbols(results: dict) -> None:
    """Per symbol: objective in-/out-of-sample, XIRR and lead over the baselines, trades; then the aggregate (#84)."""
    from tradebot.backtest.metrics import fmt

    def cell(v, f):
        return "–" if v is None else f(v)

    def lead(v):
        return f"[{'green' if v > 0 else 'red'}]{fmt('vs buy & hold', v)}[/]"

    for label, res in results.items():
        obj = res.objective
        t = Table(title=f"per symbol ({label})" if len(results) > 1 else "per symbol")
        for col in ("symbol", f"{obj} in", f"{obj} out", "XIRR out", "vs buy & hold", "vs monthly", "trades out"):
            t.add_column(col, justify="left" if col == "symbol" else "right")
        for row in res.rows():
            t.add_row(row["symbol"], f"{row['in-sample']:.3f}", cell(row["out-of-sample"], "{:.3f}".format),
                      cell(row["xirr"], lambda v: fmt("xirr", v)), cell(row["vs buy & hold"], lead),
                      cell(row["vs monthly investing"], lead), cell(row["trades"], str))
        n = len(res.symbols)
        out = res.aggregated("test")
        t.add_row(f"[bold]{res.aggregate}[/]", f"[bold]{res.aggregated('train'):.3f}[/]",
                  f"[bold]{out:.3f}[/]" if out is not None else "–", "",
                  *(f"beats on {res.beats().get(k, 0)}/{n}" for k in ("buy & hold", "monthly investing")), "")
        console.print(t)
        for s, reason in res.skipped.items():
            console.print(f"[dim]skipped {s}: {reason}[/]")
        if res.carried_by_few:
            console.print(f"[bold yellow]{label}: the result is carried by one or two symbols (median far below the "
                          "mean); it probably doesn't generalize[/]")
        if weak := {k: v for k, v in res.beats().items() if v < n / 2}:
            who = f"the {label} result" if len(results) > 1 else "the strategy"
            console.print(f"[bold red]Out-of-sample, {who} beats "
                          f"{' and '.join(f'{k} on only {v} of {n}' for k, v in weak.items())} symbols[/]")


def _print_walk_forward(wf, algorithm, symbol: str) -> None:
    """Per fold: dates, objective in/out, XIRR and lead over the baselines, trades; then the combined
    out-of-sample score, how much the parameters move between folds, the holdout and the warnings (#89)."""
    from tradebot.backtest.metrics import fmt

    def cell(v, f):
        return "–" if v is None else f(v)

    def lead(v):
        return f"[{'green' if v > 0 else 'red'}]{fmt('vs buy & hold', v)}[/]"

    def day(ts):
        return "end" if ts is None else f"{ts:%Y-%m-%d}"

    multi = any(f.result.symbols for f in wf.folds)
    obj = wf.objective
    kind = "anchored" if wf.anchored else "rolling"
    t = Table(title=f"walk-forward ({wf.method.upper()}, {kind}: {wf.train_length} train / {wf.test_length} test candles)")
    for col in ("fold", "train", "test", f"{obj} in", f"{obj} out", "XIRR out", "vs buy & hold", "vs monthly",
                "trades out"):
        t.add_column(col, justify="left" if col in ("fold", "train", "test") else "right")

    def row(f, name):
        s = f.summary()
        beats = [f" ({k}/{n})" if multi else "" for k, n in s["beats"].values()]
        t.add_row(name, f"{day(f.start)} – {day(f.split)}", f"{day(f.split)} – {day(f.end)}",
                  cell(s["in-sample"], "{:.3f}".format), cell(s["out-of-sample"], "{:.3f}".format),
                  cell(s["xirr"], lambda v: fmt("xirr", v)),
                  *(cell(s[c], lead) + b for c, b in zip(("vs buy & hold", "vs monthly investing"), beats)),
                  str(s["trades"]))

    for f in wf.folds:
        row(f, str(f.index + 1))
    n = len(wf.folds)
    counts = wf.beat_counts()
    t.add_row("[bold]all[/]", "", "", "", f"[bold]{cell(wf.mean, '{:.3f}'.format)}[/]"
              + (f" ± {wf.std:.3f}" if wf.std is not None else ""), "",
              *(f"beats in {counts[k]}/{n}" for k in ("buy & hold", "monthly investing")), "")
    if wf.holdout is not None:
        row(wf.holdout, "[bold]holdout[/]")
    console.print(t)
    if multi:
        console.print(f"[dim]{obj}: per fold the {wf.aggregate} over the symbols; XIRR and leads: mean over the "
                      "symbols, (k/n) = beaten on k of n symbols[/]")
    for s, reason in wf.skipped.items():
        console.print(f"[dim]skipped {s}: {reason}[/]")
    if stability := wf.stability():
        p = Table(title="parameters per fold")
        p.add_column("param")
        for f in wf.folds:
            p.add_column(str(f.index + 1), justify="right")
        p.add_column("moves", justify="right")
        for key, spread in stability.items():
            color = "red" if spread > 0.5 else "yellow" if spread > 0.2 else "green"
            p.add_row(key, *(f"{f.params[key]:g}" for f in wf.folds), f"[{color}]{spread:.0%}[/]")
        console.print(p)
    for w in wf.warnings():
        console.print(f"[bold red]{w}[/]")
    latest = wf.holdout or wf.folds[-1]
    console.print(f"newest parameters (tuned on the stretch {'before the holdout' if wf.holdout else 'of the last fold'})"
                  f" · reuse with: [cyan]{_reuse_command(symbol, algorithm, latest.params)}[/]")


@app.command()
def evolve(
    symbol: Annotated[Optional[str], typer.Argument(
        help="e.g. AAPL, or several separated by commas (can be left out with --watchlist)",
        autocompletion=complete.symbols)] = None,
    tf: TF = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    bars: Annotated[int, typer.Option(help="Number of candles (ignored with --start)")] = 3000,
    watchlist: Annotated[list[str], typer.Option(
        "--watchlist", "-w", help="Also evolve on this watchlist's symbols (repeatable; 'all' = every watchlist)",
        autocompletion=complete.watchlists)] = [],
    population: Annotated[int, typer.Option(help="Programs per generation")] = 300,
    generations: int = 40,
    max_depth: Annotated[Optional[int], typer.Option(
        help="Deepest nesting of a rule (default 6, with --from-scratch 12)")] = None,
    max_size: Annotated[Optional[int], typer.Option(
        help="Most building blocks (nodes) per rule (default 40, with --from-scratch 100)")] = None,
    from_scratch: Annotated[bool, typer.Option(
        "--from-scratch", help="No ready-made indicators: build rules from candles and low-level maths only, so the "
        "evolver can invent RSI, Bollinger & co. itself. Turns on the module library, deduplication, the deflated "
        "Sharpe ratio, shuffled-price runs and walk-forward folds")] = False,
    islands: Annotated[Optional[int], typer.Option(
        help="Split the population into this many islands that evolve apart and swap their best now and then "
        "(default 1, with --from-scratch 4)")] = None,
    objective: Annotated[str, typer.Option(help="sharpe | return | calmar", autocompletion=complete.choices(
        ("sharpe", ""), ("return", ""), ("calmar", "")))] = "sharpe",
    aggregate: Annotated[str, typer.Option(
        help="How the scores of all symbols and segments combine into one (mean | median | min | mean-std)",
        autocompletion=complete.choices(("mean", ""), ("median", "robust to one outlier"),
                                        ("min", "the worst one counts"), ("mean-std", "mean minus spread")))] = "mean",
    align: Annotated[bool, typer.Option(help="With several symbols: cut all of them to their common date range")] = False,
    validation: Annotated[float, typer.Option(help="Fraction of the data (after training) that picks the result")] = 0.2,
    test: Annotated[float, typer.Option(help="Fraction of the data (the newest) kept for the final, unseen score")] = 0.2,
    segments: Annotated[int, typer.Option(
        help="The training part is scored in this many consecutive stretches; a rule has to work in all of them")] = 3,
    min_trades: Annotated[int, typer.Option(help="Reject programs with fewer round trips per symbol in training")] = 4,
    parsimony: Annotated[float, typer.Option(help="Fitness penalty per building block: shorter rules win ties")] = 0.002,
    state: Annotated[bool, typer.Option(
        help="Let exit rules use the open trade (bars in trade, stop loss, take profit, trailing stop)")] = True,
    indicators: Annotated[Optional[str], typer.Option(
        help="Comma-separated indicators the rules may use (default: all, plugins included)",
        autocompletion=complete.indicators)] = None,
    permutations: Annotated[Optional[int], typer.Option(
        help="Also run the search this many times on shuffled candles, to see whether it finds as much in noise "
        "(default 0, with --from-scratch 5)")] = None,
    folds: Annotated[Optional[int], typer.Option(
        help="Walk-forward: also run the search on this many rolling windows, each scored on the stretch after it "
        "(default 0, with --from-scratch 3)")] = None,
    fee: float = 0.001,
    dca_amount: DcaAmount = None,
    seed: Annotated[Optional[int], typer.Option(help="Random seed (printed when left out, to repeat a run)")] = None,
    top: Annotated[int, typer.Option(help="Show this many of the best programs with their validation score")] = 5,
    save: Annotated[Optional[str], typer.Option(
        help="Save the result as a plugin algorithm with this name (then usable in backtest, chart, lab and bot)")] = None,
    plugin_dir: Annotated[Optional[str], typer.Option(help="Where --save writes (default: the first plugin folder)")] = None,
    force: Annotated[bool, typer.Option(help="Let --save overwrite an existing file")] = False,
):
    """Evolve a whole strategy from scratch with genetic programming: entry and exit rules built from
    prices, indicators, arithmetic and logic, with nothing written by hand.

    The data is split by date: rules evolve on the training part (scored in --segments stretches on every
    symbol), the best ones are compared on the validation part, and the winner is scored once on the
    newest --test part, next to buy & hold and monthly investing. --save writes it as a plugin."""
    from pathlib import Path

    from rich.progress import Progress

    from tradebot.core.registry import default_plugin_dirs
    from tradebot.evolve import evolve as run_evolve, valid_name
    from tradebot.optimize.base import AGGREGATES

    if aggregate not in AGGREGATES:
        raise typer.BadParameter(f"aggregate must be one of {', '.join(AGGREGATES)}", param_hint="--aggregate")
    target = None
    if save is not None:  # checked before the search, not after it
        if not valid_name(save):
            raise typer.BadParameter("lower-case letters, digits and _ only (not a Python keyword)", param_hint="--save")
        target = (Path(plugin_dir).expanduser() if plugin_dir else default_plugin_dirs()[0]) / f"{save}.py"
        if target.exists() and not force:
            raise typer.BadParameter(f"{target} exists; pick another name or add --force", param_hint="--save")
        if save in ALGORITHMS and not target.exists():
            raise typer.BadParameter(f"there already is an algorithm {save!r}; pick another name", param_hint="--save")
    names = [n.strip() for n in indicators.split(",") if n.strip()] if indicators else None
    if from_scratch and names:
        raise typer.BadParameter("from scratch uses no ready-made indicators", param_hint="--indicators")
    for n in names or ():
        if n not in INDICATORS:
            raise typer.BadParameter(f"unknown indicator {n!r}", param_hint="--indicators")
    symbols = _symbols(symbol, watchlist)
    df = load_data(symbols[0], tf, start, end, bars) if len(symbols) == 1 else _load_many(symbols, tf, start, end, bars)
    kwargs = dict(population=population, generations=generations, max_depth=max_depth, max_size=max_size,
                  objective=objective, aggregate=aggregate, align=align, validation=validation, test=test,
                  segments=segments, min_trades=min_trades, parsimony=parsimony, state=state, indicators=names,
                  fee=fee, dca_amount=dca_amount, seed=seed, from_scratch=from_scratch, islands=islands)
    with Progress(console=console) as progress:
        task = progress.add_task("evolving", total=generations)
        label = {"name": "evolving"}

        def phase(name):
            label["name"] = name
            progress.reset(task, total=generations)

        def report(gen, best, program):
            progress.update(task, advance=1, description=f"{label['name']}: generation {gen + 1} best {objective}={best:.3f}")

        try:
            res = run_evolve(df, permutations=permutations, folds=folds, progress=report, on_phase=phase, **kwargs)
        except ValueError as e:
            raise typer.BadParameter(str(e)) from None
    _print_evolution(res, tf, top)
    shown = ",".join(r.symbol for r in res.symbols) or symbols[0]
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(res.plugin_source(save))
        console.print(f"saved as [bold]{save}[/] in {target}\n"
                      f"use it with: [cyan]tradebot backtest {shown} {save}[/] · fine-tune the constants: "
                      f"[cyan]tradebot optimize {shown} {save}[/]")
    else:
        console.print(f"[dim]keep it with --save NAME (the same --seed {res.seed} repeats this run)[/]")


def _print_evolution(res, tf: str, top: int) -> None:
    """The rules, a row per symbol (objective per part, test XIRR and lead over the baselines, trades), the best
    programs with their validation score, the permutation test and the warnings (#83)."""
    from rich.panel import Panel

    from tradebot.backtest.baselines import BUY_HOLD
    from tradebot.backtest.metrics import fmt

    def cell(v, f="{:.3f}".format):
        return "–" if v is None else f(v)

    def lead(v):
        return "–" if v is None else f"[{'green' if v > 0 else 'red'}]{fmt('vs buy & hold', v)}[/]"

    def day(ts):
        return f"{ts:%Y-%m-%d}"

    console.print(Panel(str(res.program), title="evolved rules", expand=False))
    first = res.symbols[0]
    parts = [f"train {day(first.train.equity.index[0])} – {day(res.validation_start or res.test_start or first.train.equity.index[-1])}"]
    if res.validation_start is not None:
        parts.append(f"validation from {day(res.validation_start)}")
    if res.test_start is not None:
        parts.append(f"test from {day(res.test_start)}")
    unseen = res.unseen or "train"
    obj = res.objective
    t = Table(title=f"{obj} per part · {tf} · {' · '.join(parts)}")
    for col in ("symbol", "train", "validation", "test", f"XIRR {unseen}", "vs buy & hold", "vs monthly",
                f"trades {unseen}"):
        t.add_column(col, justify="left" if col == "symbol" else "right")
    for row in res.rows():
        t.add_row(row["symbol"] or "–", cell(row["train"]), cell(row["validation"]), cell(row["test"]),
                  cell(row["xirr"], lambda v: fmt("xirr", v)), lead(row["vs buy & hold"]),
                  lead(row["vs monthly investing"]), cell(row["trades"], str))
    if res.multi:
        n, beats = len(res.symbols), res.beats(unseen) if unseen != "train" else {}
        t.add_row(f"[bold]{res.aggregate}[/]", *(f"[bold]{cell(res.aggregated(p))}[/]" for p in ("train", "validation", "test")),
                  "", *(f"beats on {beats.get(k, 0)}/{n}" for k in ("buy & hold", "monthly investing")), "")
    console.print(t)
    for s, reason in res.skipped.items():
        console.print(f"[dim]skipped {s}: {reason}[/]")
    if top > 0 and len(res.hall_of_fame) > 1:
        h = Table(title="best programs in training (the one with the best validation score was picked)")
        for col in ("", "training fitness", "validation", "generation", "entry", "exit"):
            h.add_column(col, justify="left" if col in ("", "entry", "exit") else "right")
        best = sorted(res.hall_of_fame, key=lambda c: -c.fitness)
        shown = best[:top] + [c for c in best[top:] if c.program.key == res.program.key]  # the pick always shows
        for c in shown:
            mark = "[bold green]→[/]" if c.program.key == res.program.key else ""
            h.add_row(mark, f"{c.fitness:.3f}", cell(c.validation), str(c.generation + 1), *c.program.rules().values())
        console.print(h)
    console.print(f"found in generation {res.generation + 1} of {len(res.history)} · {res.evaluations} programs "
                  f"evaluated in {res.seconds:.1f} s · seed {res.seed}")
    if (pt := res.permutation) is not None:
        color = "green" if pt.p <= 0.05 else "red"
        console.print(f"permutation test: on shuffled candles the {pt.part} {obj} was "
                      f"{', '.join(f'{s:.2f}' for s in pt.shuffled)} (real: {pt.score:.2f}) · [{color}]p = {pt.p:.2f}[/]")
    if (d := res.deflated) is not None:
        color = "green" if d.significant else "red"
        console.print(f"deflated Sharpe ratio: [{color}]{d.probability:.3f}[/] (training Sharpe {d.sharpe:.3f} per candle "
                      f"vs. {d.luck:.3f} expected from luck over {d.trials} distinct strategies)")
    if (wf := res.walk_forward) is not None:
        w = Table(title=f"walk-forward: evolved afresh per window, {obj} on the unseen stretch after it")
        for col in ("fold", "window from", "unseen from", obj, "beats buy & hold", "rules"):
            w.add_column(col, justify="left" if col == "rules" else "right")
        for i, (f, score) in enumerate(zip(wf.folds, wf.scores())):
            rules = f.program.rules()
            w.add_row(str(i + 1), day(f.symbols[0].train.equity.index[0]),
                      day(f.test_start) if f.test_start is not None else "–", cell(score),
                      f"{f.beats('test')[BUY_HOLD]}/{len(f.symbols)}", f"buy: {rules['entry']} · sell: {rules['exit']}")
        console.print(w)
    for w in res.warnings():
        console.print(f"[bold red]{w}[/]")


def _make_pool(names: str | None, min_members: int, max_members: int | None):
    from tradebot.algorithms.pool import PoolEnsemble, default_pool

    if not names:
        names, skipped = default_pool()
        if skipped:
            console.print(f"[dim]left out of the pool (no default rule): {', '.join(skipped)}[/]")
    try:
        algorithm = PoolEnsemble(names.split(",") if isinstance(names, str) else names,
                                 min_members=min_members, max_members=max_members)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    console.print(f"pool: {', '.join(m.name for m in algorithm.members)}"
                  + (f" · filters: {', '.join(m.name for m in algorithm.filters)}" if algorithm.filters else ""))
    return algorithm


def _reused(algorithm, params: dict) -> dict:
    """The parameters worth showing and reusing (for a pool: those of the selected members)."""
    return algorithm.reuse_params(params) if hasattr(algorithm, "reuse_params") else params


def _reuse_command(symbol: str, algorithm, params: dict) -> str:
    flags = " ".join(f"-p {k}={v:g}" for k, v in _reused(algorithm, params).items())
    members = f" --members {','.join(algorithm.selected(params))}" if hasattr(algorithm, "selected") else ""
    return f"tradebot backtest {symbol} {algorithm.name}{members} {flags}"


def _print_members(algorithm, results: dict[str, dict]) -> None:
    """Which pool members each method selected; with two methods, whether they agree (#21)."""
    t = Table(title="pool members")
    t.add_column("member")
    t.add_column("kind")
    for name in results:
        t.add_column(name, justify="center")
    if len(results) > 1:
        t.add_column("agree", justify="center")
    chosen = [set(algorithm.selected(p)) for p in results.values()]
    for m in algorithm.pool:
        used = [m.name in c for c in chosen]
        row = ["[green]used[/]" if u else "[dim]dropped[/]" for u in used]
        if len(results) > 1:
            row.append("[green]yes[/]" if len(set(used)) == 1 else "[red]no[/]")
        t.add_row(m.name, "filter" if m in algorithm.filters else "strategy", *row)
    console.print(t)


def _print_comparison(cmp, symbol: str, algorithm) -> None:
    if hasattr(algorithm, "selected"):
        _print_members(algorithm, {"GA": cmp.ga.best_params, "PSO": cmp.pso.best_params})
    t = Table(title="GA vs PSO: parameters")
    for col in ("param", "GA", "PSO", "difference (% of range)"):
        t.add_column(col, justify="left" if col == "param" else "right")
    for k, d in cmp.param_diff.items():
        if k.startswith("use."):  # shown in the members table
            continue
        color = "green" if d <= 0.10 else "yellow" if d <= 0.25 else "red"
        t.add_row(k, f"{cmp.ga.best_params[k]:g}", f"{cmp.pso.best_params[k]:g}", f"[{color}]{d:.0%}[/]")
    console.print(t)
    if cmp.ga.symbols:
        _print_symbols({"GA": cmp.ga, "PSO": cmp.pso})
        console.print(f"evaluations: GA {cmp.ga.evaluations} · PSO {cmp.pso.evaluations}")
    else:
        _print_comparison_results(cmp)
    color = {"consistent": "green", "fragile": "yellow", "divergent": "red"}[cmp.verdict]
    console.print(f"agreement [bold]{cmp.agreement:.0%}[/] · max parameter difference {cmp.max_param_diff:.0%} · "
                  f"in-sample {cmp.objective} gap {cmp.fitness_gap:.1%}")
    console.print(f"verdict: [bold {color}]{cmp.verdict.upper()}[/] - {cmp.explanation()}")
    best = cmp.better
    console.print(f"better in-sample: {best.method.upper()} · reuse with: "
                  f"[cyan]{_reuse_command(symbol, algorithm, best.best_params)}[/]")


def _print_comparison_results(cmp) -> None:
    columns, bases = {}, {}
    for period, attr in (("in-sample", "train"), ("out-of-sample", "test")):
        for r in (cmp.ga, cmp.pso):
            bl, res = getattr(r, f"{attr}_baselines"), getattr(r, attr)
            if res is not None:
                columns[f"{r.method.upper()} {period}"] = bl.columns(r.method, res)[r.method]
                bases.update({f"{k} {period}": v.metrics for k, v in bl.results.items()})  # same split for both
    m = results_table("GA vs PSO: results", columns)
    m.add_row("evaluations", str(cmp.ga.evaluations), str(cmp.pso.evaluations), *[""] * (len(columns) - 2))
    console.print(m)
    console.print(results_table("baselines", bases))
    for r in (cmp.ga, cmp.pso):
        warn_not_beaten(r.test_baselines, "Out-of-sample", f"the {r.method.upper()} result")


def confirm_live_risk(live: bool) -> None:
    """Ask once for the live-trading risk acceptance, then remember it in config.toml (#58)."""
    from tradebot import legal

    if live and not legal.live_risk_accepted():
        console.print(legal.LIVE_RISK, highlight=False)
        typer.confirm("Do you understand and accept this?", abort=True)
        if not legal.accept_live_risk():
            console.print("[yellow]couldn't save the acceptance to the config file; you'll be asked again next time[/]")


@bot_app.command("run")
def bot_run(
    symbol: Symbol,
    algo: Algo,
    broker: Broker = "binance",
    tf: TF = "1h",
    stake: Annotated[float, typer.Option(help="Max quote-currency value per entry")] = 100.0,
    param: Params = [],
    members: Members = None,
    live: Annotated[bool, typer.Option(help="Actually place orders (default: dry run)")] = False,
    sandbox: Annotated[bool, typer.Option(help="Use the exchange testnet (ccxt brokers)")] = False,
    once: Annotated[bool, typer.Option(help="Evaluate a single candle and exit")] = False,
):
    """Run an algorithm live against a broker (dry run unless --live)."""
    from tradebot.live import TradingBot

    confirm_live_risk(live)

    kwargs = {"sandbox": True} if sandbox else {}
    b = create_broker(broker, **kwargs)
    if live and not b.can_trade:
        raise typer.BadParameter(f"{broker} is read-only")
    bot = TradingBot(b, make_algo(algo, param, members), symbol, tf, stake, dry_run=not live,
                     on_event=lambda e: console.print(
                         f"[dim]{e.time:%Y-%m-%d %H:%M}[/] {symbol} target={e.target:.2f} "
                         f"price={e.price:.6g} → [bold]{e.action}[/] {e.amount:.6g}"
                         + (" [yellow](dry run)[/]" if e.dry_run else "")
                         + (f" order {e.order.id} {e.order.status.value}" if e.order else "")))
    mode = "[red]LIVE[/]" if live else "[yellow]dry run[/]"
    console.print(f"bot {algo} on {broker}:{symbol} {tf}, stake {stake} · {mode} · {b.execution_mode.value}")
    try:
        bot.run(max_steps=1 if once else None)
    except KeyboardInterrupt:
        console.print("stopped")


@bot_app.command("pair-run")
def bot_pair_run(
    leg_a: Annotated[str, typer.Argument(help="First leg, e.g. BTC/USDT:USDT")],
    leg_b: Annotated[str, typer.Argument(help="Second leg, e.g. ETH/USDT:USDT")],
    broker: Broker = "binance",
    tf: TF = "1h",
    stake: Annotated[float, typer.Option(help="Quote-currency value of both legs together per entry")] = 100.0,
    window: Annotated[int, typer.Option(help="Candles for the rolling hedge ratio and z-score")] = 60,
    entry: Annotated[float, typer.Option(help="Open when the gap is this many σ wide")] = 2.0,
    exit: Annotated[float, typer.Option(help="Close when it is back inside this many σ")] = 0.5,
    stop: Annotated[float, typer.Option(help="Give up beyond this many σ")] = 4.0,
    method: Annotated[str, typer.Option(help="Hedge ratio: rolling or kalman")] = "rolling",
    live: Annotated[bool, typer.Option(help="Actually place orders (default: dry run)")] = False,
    sandbox: Annotated[bool, typer.Option(help="Use the exchange testnet (ccxt brokers)")] = False,
    once: Annotated[bool, typer.Option(help="Evaluate a single candle and exit")] = False,
):
    """Trade the gap of two assets live: long one and short the other (dry run unless --live).

    The short leg is a plain sell, so the broker needs margin or futures for it: give the legs as symbols it
    can short. The bot doesn't remember an open pair across a restart."""
    from tradebot.live import PairBot

    legs, symbol = (leg_a, leg_b), f"{leg_a} / {leg_b}"
    confirm_live_risk(live)
    b = create_broker(broker, **({"sandbox": True} if sandbox else {}))
    if live and not b.can_trade:
        raise typer.BadParameter(f"{broker} is read-only")
    try:
        bot = PairBot(b, legs, tf, stake, window, entry, exit, stop, method, dry_run=not live,
                      on_event=lambda e: console.print(
                          f"[dim]{e.time:%Y-%m-%d %H:%M}[/] {e.pair} z={e.z:+.2f} β={e.beta:.2f} → [bold]{e.action}[/]"
                          + (f" {e.amount_a:.6g} {legs[0]} / {e.amount_b:.6g} {legs[1]}" if e.action != "hold" else "")
                          + (" [yellow](dry run)[/]" if e.dry_run else "")))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    mode = "[red]LIVE[/]" if live else "[yellow]dry run[/]"
    console.print(f"pair bot {symbol} on {broker} {tf}, stake {stake} · {mode} · {b.execution_mode.value}")
    try:
        bot.run(max_steps=1 if once else None)
    except KeyboardInterrupt:
        console.print("stopped")


@app.command()
def positions(broker: Broker = "binance"):
    """Show balances / open positions at a broker."""
    b = create_broker(broker)
    t = Table(title=f"{broker} balances")
    for col in ("asset", "free", "used", "total"):
        t.add_column(col, justify="right" if col != "asset" else "left")
    for asset, bal in sorted(b.get_balance().items()):
        t.add_row(asset, f"{bal.free:g}", f"{bal.used:g}", f"{bal.total:g}")
    console.print(t)


@app.command()
def portfolio(broker: Annotated[list[str], typer.Option(help="Broker(s) to aggregate",
                                                       autocompletion=complete.brokers)] = ["scalable"]):
    """Portfolio value, allocation and dividend metrics (current yield, annual yield, yield on cost)."""
    from tradebot.data import MarketDataEngine
    from tradebot.portfolio import build_portfolio

    engine = MarketDataEngine()
    positions_ = [p for name in broker for p in create_broker(name, data=engine).get_positions()]
    with console.status("pricing holdings ..."):
        pf = build_portfolio(positions_, engine.latest_price)
    t = Table(title="portfolio")
    cols = ["symbol", "quantity", "price", "value", "weight", "current yield", "annual yield", "YoC", "income/yr"]
    for c in cols:
        t.add_column(c, justify="left" if c == "symbol" else "right")
    pct = lambda v: "–" if v is None else f"{v * 100:.2f}%"  # noqa: E731
    for h in sorted(pf.holdings, key=lambda h: -h.value):
        t.add_row(h.symbol, f"{h.quantity:g}", f"{h.price:,.2f}", f"{h.value:,.2f}", pct(h.value / pf.total_value),
                  pct(h.current_yield), pct(h.annual_yield), pct(h.yield_on_cost), f"{h.annual_income:,.2f}")
    console.print(t)
    console.print(f"total [bold]{pf.total_value:,.2f}[/] · income/yr {pf.annual_income:,.2f} · "
                  f"yield {pct(pf.current_yield)} · trailing {pct(pf.annual_yield)} · YoC {pct(pf.yield_on_cost)}")


@app.command()
def screen(
    symbols: Annotated[Optional[list[str]], typer.Argument(help="Defaults to the watchlists",
                                                           autocompletion=complete.symbols)] = None,
    filter: Annotated[list[str], typer.Option("--filter", "-f", help='e.g. -f "rsi < 30" -f "close > sma200"')] = [],
    sort: str = "change_1d",
    watchlist: Annotated[Optional[str], typer.Option("--watchlist", "-w", help=WATCHLIST_HELP)] = None,
):
    """Screen symbols by technicals, volume and market cap."""
    from tradebot.data import MarketDataEngine
    from tradebot.screener import run_screen

    symbols = symbols or _watchlist(watchlist)
    engine = MarketDataEngine()
    with console.status("screening ..."):
        table = run_screen(symbols, lambda s: engine.get_ohlc(s, "1d", start=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=420)), filter)
    table = table.sort_values(sort, ascending=False)
    t = Table(title=f"screener ({len(table)} matches)")
    for c in table.columns:
        t.add_column(c, justify="left" if c == "symbol" else "right")
    for _, row in table.iterrows():
        t.add_row(*[_fmt_cell(c, row[c]) for c in table.columns])
    console.print(t)


def parse_where(text: str) -> dict:
    """``"fast > slow"`` / ``"rsi <= 30"`` -> a scanner filter."""
    import re

    m = re.fullmatch(r"\s*(\w+)\s*(>=|<=|==|!=|>|<)\s*(\S+)\s*", text)
    if not m:
        raise typer.BadParameter(f"{text!r}: use '<output> <op> <number|output>', e.g. 'fast > slow'", param_hint="--where")
    output, op, value = m.groups()
    try:
        return {"output": output, "op": op, "value": float(value)}
    except ValueError:
        return {"output": output, "op": op, "value": value}


@app.command()
def scan(
    symbols: Annotated[Optional[list[str]], typer.Argument(help="Defaults to the watchlists",
                                                           autocompletion=complete.symbols)] = None,
    indicator: Annotated[str, typer.Option("--indicator", "-i", help="Any scannable indicator",
                                           autocompletion=complete.scannable)] = "ma_cross",
    param: Annotated[list[str], typer.Option("--param", "-p", help="e.g. -p fast=9 -p slow=21")] = [],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", autocompletion=complete.timeframes,
                                           help="1m 5m 15m 30m 1h 2h 4h 1d 1w 1mo")] = "1d",
    where: Annotated[list[str], typer.Option("--where", help='e.g. --where "fast > slow" --where "rsi < 30"')] = [],
    alert: Annotated[list[str], typer.Option("--alert", help="Alert that must be on; !NAME: off. e.g. --alert cross_up")] = [],
    watchlist: Annotated[Optional[str], typer.Option("--watchlist", "-w", help=WATCHLIST_HELP)] = None,
):
    """Scan symbols with one indicator, filtered on its values and alerts (the Scanner tab's engine)."""
    from tradebot import scanner

    params = {}
    for item in param:
        key, sep, value = item.partition("=")
        try:
            params[key] = float(value)
        except ValueError:
            raise typer.BadParameter(f"{item!r}: use name=number", param_hint="--param") from None
    filters = [parse_where(w) for w in where] + [
        {"alert": a.lstrip("!"), "state": not a.startswith("!")} for a in alert]
    symbols = symbols or _watchlist(watchlist)
    try:
        with console.status("scanning ..."):
            result = scanner.scan(symbols, indicator, params, timeframe, filters)
    except (ValueError, KeyError, scanner.ScanBusy) as exc:
        raise typer.BadParameter(str(exc.args[0] if exc.args else exc)) from None
    t = Table(title=f"scan: {indicator} {timeframe}")
    for c in result.table.columns:
        t.add_column(c, justify="left" if c == "symbol" else "right")
    for _, row in result.table.iterrows():
        t.add_row(*[str(row[c]) if isinstance(row[c], (bool, np.bool_)) else _fmt_cell(c, row[c]) for c in result.table.columns])
    console.print(t)
    console.print(result.summary())
    for sym, why in list(result.skipped.items())[:5]:
        console.print(f"[dim]skipped {sym}: {why}[/]")


WATCHLIST_HELP = "Without symbols: only this watchlist's (default: all watchlists)"


def _watchlist(name: str | None) -> list[str]:
    from tradebot.config import watchlist

    try:
        return watchlist(name)
    except KeyError as e:
        raise typer.BadParameter(e.args[0], param_hint="--watchlist") from None


def _fmt_cell(col, v):
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "–"
    if col.startswith("change") or col == "atr_pct":
        color = "green" if v > 0 else "red"
        return f"[{color}]{v * 100:+.2f}%[/]" if col.startswith("change") else f"{v * 100:.2f}%"
    if col == "market_cap":
        return f"{v / 1e9:,.1f}B"
    return f"{v:,.2f}"


@app.command()
def news(symbols: Annotated[Optional[list[str]], typer.Argument(autocompletion=complete.symbols)] = None, limit: int = 25,
         watchlist: Annotated[Optional[str], typer.Option("--watchlist", "-w", help=WATCHLIST_HELP)] = None):
    """Latest headlines for symbols (defaults to the watchlists)."""
    from tradebot.news import fetch_news

    symbols = symbols or _watchlist(watchlist)
    with console.status("fetching news ..."):
        items = fetch_news(symbols, limit)
    for n in items:
        console.print(f"[dim]{n.published:%Y-%m-%d %H:%M}[/] [cyan]{n.symbol:>9}[/] [bold]{n.title}[/] "
                      f"[dim]({n.publisher})[/]\n           [link={n.url}]{n.url}[/link]")


@app.command()
def pair(symbol: Annotated[str, typer.Argument(help="Two symbols joined by a colon: KO:PEP, BTC/USDT:ETH/USDT")],
         tf: Annotated[str, typer.Option(help="Timeframe")] = "1d",
         bars: Annotated[int, typer.Option(help="Candles to analyse")] = 500):
    """How far two assets that move together are apart now: z-score, hedge ratio, half-life.

    The pair itself is also a symbol (``chart``, ``backtest``, ... accept ``KO:PEP``): the candles are the
    ratio of the first to the second, so everything that works on one symbol works on the gap."""
    from tradebot.analysis.pairs import pair_stats
    from tradebot.core.models import split_pair

    legs = split_pair(symbol)
    if legs is None:
        raise typer.BadParameter("write the pair as A:B, e.g. KO:PEP")
    a, b = (load_data(leg, tf, None, None, bars)["close"] for leg in legs)
    try:
        st = pair_stats(a, b)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    t = Table(title=f"{legs[0]} vs {legs[1]} · {tf} · {st.bars} common candles")
    t.add_column("measure")
    t.add_column("value", justify="right")
    t.add_row("correlation of returns", f"{st.correlation:.2f}")
    t.add_row("hedge ratio β (log a ≈ α + β·log b)", f"{st.beta:.2f}")
    t.add_row("z-score of the gap now", f"{st.zscore:+.2f}")
    t.add_row("half-life of a gap", "does not revert" if st.half_life is None else f"{st.half_life:.1f} candles")
    from tradebot.analysis.pairs import cointegration, pair_health

    co = cointegration(a, b)
    t.add_row("cointegration (Engle-Granger)", f"t = {co.stat:.2f}, {co.p_text}" + (" ✓" if co.passed else ""))
    console.print(t)
    try:
        health = pair_health(a, b)
    except ValueError:
        return
    if health.broken:
        console.print("[bold red]this pair may have stopped working:[/] " + "; ".join(health.reasons))
    else:
        console.print("[green]the last 60 candles still look like the history before them[/]")


@app.command("pair-backtest")
def pair_backtest_(
        symbol: Annotated[str, typer.Argument(help="A:B, e.g. KO:PEP: long one leg and short the other")],
        tf: Annotated[str, typer.Option(help="Timeframe")] = "1d",
        bars: Annotated[int, typer.Option(help="Candles")] = 1500,
        window: Annotated[int, typer.Option(help="Candles for the rolling hedge ratio and z-score")] = 60,
        entry: Annotated[float, typer.Option(help="Open when the gap is this many σ wide")] = 2.0,
        exit: Annotated[float, typer.Option(help="Close when it is back inside this many σ")] = 0.5,
        stop: Annotated[float, typer.Option(help="Give up beyond this many σ")] = 4.0,
        fee: Annotated[float, typer.Option(help="Fee per change of either leg")] = 0.001,
        borrow: Annotated[float, typer.Option(help="Yearly cost of the short leg (stocks ~0.02, crypto 0)")] = 0.02,
        method: Annotated[str, typer.Option(help="Hedge ratio: rolling (OLS over --window) or kalman (adaptive filter)",
                                            autocompletion=complete.choices(("rolling", "rolling regression"),
                                                                            ("kalman", "Kalman filter")))] = "rolling"):
    """Backtest trading the gap of two assets: long A and short B (or the reverse) at the same time."""
    from tradebot.analysis.pairs import pair_health
    from tradebot.backtest.pairs import pair_backtest
    from tradebot.core.models import split_pair

    legs = split_pair(symbol)
    if legs is None:
        raise typer.BadParameter("write the pair as A:B, e.g. KO:PEP")
    a, b = (load_data(leg, tf, None, None, bars) for leg in legs)
    try:
        res = pair_backtest(a, b, window, entry, exit, stop, fee, borrow, method=method)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    console.print(results_table(f"{symbol} · {tf} · long/short", {"pairs": res.metrics}))
    try:
        health = pair_health(a["close"], b["close"])
        if health.broken:
            console.print("[bold red]this pair may have stopped working:[/] " + "; ".join(health.reasons))
    except ValueError:
        pass


@app.command("pair-scan")
def pair_scan(symbols: Annotated[list[str], typer.Argument(help="Symbols to search for pairs, e.g. KO PEP PG MDLZ")],
              tf: Annotated[str, typer.Option(help="Timeframe")] = "1d",
              bars: Annotated[int, typer.Option(help="Candles")] = 1000,
              split: Annotated[float, typer.Option(help="Share of the history pairs are picked on; the rest confirms")] = 0.6,
              min_correlation: Annotated[float, typer.Option(help="Pre-filter: return correlation on the first part")] = 0.7,
              correction: Annotated[str, typer.Option(
                  help="Allowance for how many pairs were tested: fdr | bonferroni | none",
                  autocompletion=complete.choices(("fdr", "about 5 % of the pairs kept are false"),
                                                  ("bonferroni", "at most 5 % chance of any false pair"),
                                                  ("none", "no correction (many chance finds in big scans)")))] = "fdr",
              same_sector: Annotated[bool, typer.Option(
                  help="Pre-filter: only pair stocks Yahoo lists in the same sector (cuts the pairs to test)")] = False,
              backend: Annotated[str, typer.Option(
                  help="Where the tests run: auto (GPU if available) | gpu | cuda | rocm | cpu | numpy",
                  autocompletion=complete.choices(("auto", "GPU if available"), ("gpu", "any GPU"), ("cuda", "NVIDIA only"),
                                                  ("rocm", "AMD only (alias: amd)"), ("cpu", "torch on the CPU"), ("numpy", "")))] = "auto"):
    """Find pairs that move together: picked on the first part of the history, confirmed on the rest."""
    from tradebot.analysis.pairs import find_pairs

    if len(symbols) < 2:
        raise typer.BadParameter("give at least two symbols")
    closes = {}
    for sym in symbols:
        try:
            closes[sym] = load_data(sym, tf, None, None, bars)["close"]
        except Exception as e:  # one dead symbol shouldn't end the scan
            console.print(f"[yellow]skipping {sym}: {e}[/]")
    try:
        groups = None
        if same_sector:
            from tradebot.data.providers import sectors

            with console.status("looking up sectors ..."):
                groups = sectors(list(closes))
            console.print(f"sector known for {len(groups)} of {len(closes)} symbols; the others aren't paired")
        scan = find_pairs(closes, split, min_correlation, correction=correction, groups=groups, backend=backend)
    except (ValueError, RuntimeError) as e:  # RuntimeError: a GPU was asked for but there is none
        raise typer.BadParameter(str(e)) from None
    console.print(f"{scan.tested} pairs tested ({scan.skipped} dropped by the correlation filter), "
                  f"{scan.passed_train} passed the test on the first {split:.0%} (correction: {correction}; "
                  f"without it about {scan.expected_by_chance:.1f} would pass by chance alone). [bold]{len(scan.found)}[/] held up on the rest."
                  f" [dim]({scan.backend})[/]")
    if not scan.found:
        return
    t = Table(title="Pairs confirmed on data they weren't picked on")
    for col in ("pair", "correlation", "β", "t (picked)", "t (confirmed)", "half-life", "z now"):
        t.add_column(col, justify="left" if col == "pair" else "right")
    for c in scan.found:
        t.add_row(c.symbol, f"{c.correlation:.2f}", f"{c.beta:.2f}", f"{c.train.stat:.2f}", f"{c.test.stat:.2f}",
                  f"{c.half_life:.1f}", f"{c.zscore:+.2f}")
    console.print(t)


@app.command()
def halving(live: Annotated[bool, typer.Option(
        help="Also forecast from the current chain tip's pace (asks the Blockstream API)")] = False):
    """Past Bitcoin halvings, how far each was from the 4-year rule, and the forecast of the next ones."""
    from tradebot.analysis import halving as hv

    def day(ts):
        return f"{pd.Timestamp(ts, unit='s', tz='UTC'):%Y-%m-%d %H:%M}"

    t = Table(title=f"Bitcoin halvings · 4-year rule: {hv.INTERVAL:,} blocks × {hv.TARGET_SPACING} s = "
                    f"{hv.NOMINAL_DAYS:.2f} days")
    for col in ("block", "date (UTC)", "days", "vs 4-year rule", "s / block"):
        t.add_column(col, justify="left" if col == "date (UTC)" else "right")
    t.add_row("0", day(hv.GENESIS), "–", "–", "–")
    for (height, ts), prev, pace in zip(hv.HALVINGS, hv.STARTS, hv.spacings(hv.STARTS)):
        days = (ts - prev) / hv.DAY
        t.add_row(f"{height:,}", day(ts), f"{days:.1f}", f"{days / hv.NOMINAL_DAYS - 1:+.1%}", f"{pace:.1f}")
    console.print(t)

    best = hv.best_estimator()
    t = Table(title="Forecasts: each past halving predicted from the ones before it")
    t.add_column("estimator", style="bold cyan")
    for f in hv.backtest(best):
        t.add_column(f"error {f.height:,}", justify="right")
    t.add_column("mean |error|", justify="right")
    t.add_column(f"next ({hv.HALVINGS[-1][0] + hv.INTERVAL:,})")
    for name in hv.ESTIMATORS:
        mark = " ✓" if name == best else ""
        t.add_row(name + mark, *(f"{f.error_days:+.0f} d" for f in hv.backtest(name)),
                  f"{hv.mean_abs_error(name):.0f} d", day(hv.predict(estimator=name)[0])[:10])
    console.print(t)
    later = hv.predict(count=4)[1:]
    console.print(f"Used by the [bold]halving[/] indicator and [bold]halving_cycle[/]: {best} ✓. Later halvings: "
                  + ", ".join(f"{day(ts)[:10]}" for ts in later))
    if live:
        try:
            height, ts = hv.fetch_tip()
        except Exception as exc:  # offline, API down, ...
            raise typer.BadParameter(f"couldn't reach the Blockstream API ({exc})")
        target, when, pace = hv.next_from_tip(height, ts)
        console.print(f"Chain tip {height:,} at {day(ts)}: {pace:.1f} s per block this epoch → block {target:,} "
                      f"around [bold]{day(when)}[/] UTC")


@app.command()
def device():
    """Show which compute backend the optimizer will use."""
    from tradebot.optimize import select_backend

    console.print(select_backend().description)


@app.command()
def gui():
    """Launch the desktop application."""
    try:
        from tradebot.gui.app import main as gui_main
    except ImportError as exc:
        raise typer.BadParameter(f"GUI dependencies missing ({exc}); install with: pip install 'tradebot[gui]'")
    gui_main()


if __name__ == "__main__":
    app()


@cache_app.command("info")
def cache_info():
    """Where the cache is, how big it is and what it holds."""
    from tradebot.data.cache import DiskCache, human_size

    cache = DiskCache.from_config()
    info = (cache or DiskCache()).info()
    t = Table(show_header=False)
    t.add_row("folder", str(info.folder))
    t.add_row("status", f"on, at most {cache.max_bytes // 2**20:,} MB" if cache else "off ([cache] enabled = false)")
    t.add_row("chart data", f"{human_size(info.ohlc_bytes)} in {info.series} series")
    t.add_row("symbol lists", human_size(info.symbol_bytes))
    t.add_row("total", human_size(info.total_bytes))
    console.print(t)


@cache_app.command("clear")
def cache_clear(
    ohlc: Annotated[bool, typer.Option("--ohlc", help="Only the chart data (candles)")] = False,
    symbols: Annotated[bool, typer.Option("--symbols", help="Only the symbol lists")] = False,
):
    """Delete cached data (everything unless --ohlc or --symbols); it is downloaded again when needed."""
    from tradebot.data.cache import DiskCache, human_size

    cache = DiskCache()
    before = cache.info().total_bytes
    both = not (ohlc or symbols)
    cache.clear(ohlc=ohlc or both, symbols=symbols or both)
    console.print(f"deleted {human_size(before - cache.info().total_bytes)}")
