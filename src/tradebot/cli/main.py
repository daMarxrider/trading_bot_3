# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""``tradebot`` command line interface."""

from __future__ import annotations

import logging
from typing import Annotated, Optional

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
    backend: Annotated[str, typer.Option(help="auto | gpu | cpu | numpy", autocompletion=complete.choices(
        ("auto", "GPU if available"), ("gpu", ""), ("cpu", "torch on the CPU"), ("numpy", "")))] = "auto",
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
):
    """Optimize algorithm parameters with a genetic algorithm and/or particle swarm (GPU accelerated
    when available) and report out-of-sample results. ``--method both`` also measures how much the two
    methods disagree - large disagreement means a fragile optimum.

    Without an algorithm, the optimizer also picks *which* strategies vote and which indicators filter
    the entries, from all of them or from ``--pool``.

    With several symbols (comma-separated, or ``--watchlist``) one parameter set is scored on every
    symbol and the scores are combined (``--aggregate``): settings that only work on one chart lose."""
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
                  dca_amount=dca_amount, aggregate=aggregate, align=align)
    n_genes = len([k for k in algorithm.param_space() if k not in kwargs["fixed"]])
    console.print(f"backend: [bold]{kwargs['backend'].description}[/] · {size} · {n_genes} parameters")
    symbol = symbols[0]
    with Progress(console=console) as progress:
        task = progress.add_task("optimizing", total=generations)
        label = {"name": method.upper()}

        def phase(name):
            label["name"] = name.upper()
            progress.reset(task, total=generations)

        def report(step, best, params):
            progress.update(task, advance=1, description=f"{label['name']} step {step + 1} best {objective}={best:.3f}")

        try:
            if method == "both":
                cmp = run_comparison(algorithm, df, progress=report, on_phase=phase, **kwargs)
            else:
                res = OPTIMIZERS[method](algorithm, df, **kwargs).run(report)
        except ValueError as e:  # e.g. no symbol with enough history
            raise typer.BadParameter(str(e)) from None
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
    from tradebot import legal
    from tradebot.live import TradingBot

    if live and not legal.live_risk_accepted():  # asked once, then remembered in config.toml (#58)
        console.print(legal.LIVE_RISK, highlight=False)
        typer.confirm("Do you understand and accept this?", abort=True)
        if not legal.accept_live_risk():
            console.print("[yellow]couldn't save the acceptance to the config file; you'll be asked again next time[/]")

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
