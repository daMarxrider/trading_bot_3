# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""``tradebot`` command line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.markup import escape
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
combo_app = typer.Typer(help="Saved strategy combos (combos.toml): entry and exit rules built from strategies and "
                             "indicator conditions.", no_args_is_help=True)
app.add_typer(combo_app, name="combo")
notify_app = typer.Typer(help="Notification channels ([notify.*] in config.toml): Telegram, Discord, webhook, "
                              "email, desktop.", no_args_is_help=True)
app.add_typer(notify_app, name="notify")
console = Console()

Params = Annotated[list[str], typer.Option("--param", "-p", help="Algorithm parameter, e.g. -p lower=30",
                                           autocompletion=complete.algorithm_params)]
Members = Annotated[Optional[str], typer.Option(
    help="Comma-separated members for the 'vote' ensemble, or for the 'pool' (strategies and indicators)",
    autocompletion=complete.members)]
Entry = Annotated[Optional[str], typer.Option(
    help="Combo entry rule (with the 'combo' strategy), e.g. \"elliott_wave & close>sma(200)\" or "
         "\"weighted(rsi_threshold:0.5, macd_cross:0.5) >= 0.6\"")]
Exit = Annotated[Optional[str], typer.Option(
    help="Combo exit rule, e.g. \"macd_cross | stop_atr(2)\" (default: when the entry rule stops holding)")]
Fills = Annotated[Optional[str], typer.Option(
    help="Combo fills: close (stops, take-profits and limit entries checked on the close) or intrabar (filled "
         "inside the candle against its high/low, at the open on a gap; stop first when both are hit)",
    autocompletion=complete.choices(("close", "fill at the close"), ("intrabar", "fill against high/low"))
)]
Sizing = Annotated[Optional[str], typer.Option(
    help="Position sizing on top of the strategy (#147): fixed (fraction of the capital), vol (volatility "
         "target), risk (fixed risk per trade, the ATR stop distance decides the size) or kelly (fractional "
         "Kelly). Tune it with -p size.<name>=value, e.g. -p size.target_vol=0.15",
    autocompletion=complete.choices(("none", "full size"), ("fixed", "fixed fraction"),
                                    ("vol", "volatility target"), ("risk", "fixed risk per trade"),
                                    ("kelly", "fractional Kelly"))
)]
Direction = Annotated[Optional[str], typer.Option(
    help="Trade direction (#149): long (default), short (short on the strategy's sell signal) or long_short "
         "(long on its buy signal, short on its sell signal). Shorts are backtested with liquidation and "
         "--borrow-fee / --funding; the live bot does not open shorts",
    autocompletion=complete.choices(("long", "long only"), ("short", "short only"),
                                    ("long_short", "long and short")))]
Leverage = Annotated[float, typer.Option(
    min=1.0, max=125.0, help="Leverage (#149): each position is this many times its margin. Positions are "
                             "liquidated (the margin is lost) when the price reaches the liquidation price")]
MaintenanceMargin = Annotated[float, typer.Option(
    help="Maintenance margin rate, fraction of the notional (0.005 = 0.5%); liquidation below it")]
BorrowFee = Annotated[float, typer.Option(
    help="Yearly borrow fee of a short, fraction (stocks ~0.01-0.05, perpetuals 0)")]
Funding = Annotated[Optional[str], typer.Option(
    help="Funding of a perpetual future: a rate per 8 hours (0.0001 = 0.01%, longs pay shorts when positive) or "
         "'history' for the exchange's historical rates (crypto)")]
TF = Annotated[str, typer.Option("--tf", help="Timeframe: 1m 5m 15m 30m 1h 4h 1d 1w", autocompletion=complete.timeframes)]
Symbol = Annotated[str, typer.Argument(help="e.g. AAPL, SAP.DE, BTC/USDT", autocompletion=complete.symbols)]
Algo = Annotated[str, typer.Argument(help="Algorithm name (tradebot list)", autocompletion=complete.algorithms)]
Broker = Annotated[str, typer.Option(help="Registered broker name", autocompletion=complete.brokers)]
Slippage = Annotated[Optional[str], typer.Option(
    help="Slippage per side in basis points (5 = 0.05%), charged on top of the fee; 'auto' = suggested for the "
    "asset class (crypto 5, stock 2). Default: [backtest] slippage_bps in config.toml, else 0")]
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
    from tradebot.algorithms.combo import load_combos

    load_combos()


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
    """``name`` or ``name:k=v,k=v`` -> indicator instance; ``tf=1d`` computes it on a higher timeframe (#161)."""
    name, _, args = spec.partition(":")
    items = [a for a in args.split(",") if a]
    tf = [a.partition("=")[2].strip() for a in items if a.partition("=")[0].strip() == "tf"]
    items = [a for a in items if a.partition("=")[0].strip() != "tf"]
    try:
        return INDICATORS.get(name)(tf=tf[-1] if tf else None, **parse_kv(items))
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--indicator") from None


def _higher_timeframes(text: str | None, base: str) -> tuple[str, ...]:
    """``"1d,1w"`` -> ``("1d", "1w")``, each a known timeframe longer than ``base`` (#161)."""
    from tradebot.core import mtf

    out = tuple(dict.fromkeys(t.strip() for t in (text or "").split(",") if t.strip()))
    for t in out:
        try:
            mtf.check(t)
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="--higher-tf") from None
        if base in mtf.TIMEFRAMES and mtf.seconds(t) <= mtf.seconds(base):
            raise typer.BadParameter(f"{t} is not longer than the candles ({base})", param_hint="--higher-tf")
    return out


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


def make_algo(name: str, params: list[str], members: str | None, entry: str | None = None, exit: str | None = None,
              fills: str | None = None, sizing: str | None = None, direction: str | None = None):
    from tradebot.algorithms.direction import directed
    from tradebot.algorithms.sizing import PREFIX, sized

    try:
        given = parse_kv(params)
        size = {k: v for k, v in given.items() if k.startswith(PREFIX)}
        algo = build_algorithm(name, {k: v for k, v in given.items() if k not in size},
                               members.split(",") if members else None, entry, exit, fills)
        return sized(directed(algo, direction), sizing, **size)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None


def _margin(leverage: float = 1.0, maintenance: float = 0.005, borrow: float = 0.0, funding: str | None = None):
    """The backtest's :class:`Margin` from the CLI options (#149), None when they are all at their defaults."""
    from tradebot.backtest.margin import Margin

    rate = 0.0 if funding in (None, "history") else _float(funding, "--funding")
    if leverage == 1.0 and maintenance == 0.005 and not borrow and not rate:
        return None
    try:
        return Margin(leverage, maintenance, borrow, rate)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None


def _float(text: str, hint: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise typer.BadParameter(f"expected a number or 'history', got {text!r}", param_hint=hint) from None


def _funding_history(funding: str | None, symbol: str, df: pd.DataFrame):
    """Historical funding per candle of ``df`` for ``--funding history``, else None."""
    if funding != "history":
        return None
    from tradebot.data.funding import fetch_history, per_candle

    try:
        with console.status(f"fetching funding rates of {symbol} ..."):
            rates = fetch_history(symbol, df.index[0], df.index[-1] + (df.index[-1] - df.index[-2]))
    except Exception as e:  # noqa: BLE001 - no perpetual, offline, exchange error
        raise typer.BadParameter(f"no funding history for {symbol}: {e}", param_hint="--funding") from None
    return per_candle(rates, df.index)


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
    indicator: Annotated[list[str], typer.Option("--indicator", "-i", help="e.g. -i sma:period=50 -i rsi; "
                                                 "tf= on a higher timeframe: -i sma:period=200,tf=1d",
                                                 autocompletion=complete.indicator_specs)] = [],
    algo: Annotated[Optional[str], typer.Option(help="Overlay this algorithm's buy/sell signals",
                                                autocompletion=complete.algorithms)] = None,
    param: Params = [],
    members: Members = None,
    entry: Entry = None,
    exit: Exit = None,
    fills: Fills = None,
    sizing: Sizing = None,
    direction: Direction = None,
    line: Annotated[bool, typer.Option(help="Line chart instead of candles")] = False,
    volume: Annotated[bool, typer.Option(help="Volume bars under the candles (when the data has volume)")] = True,
    kind: Annotated[str, typer.Option("--type", help="candles | line | heikin-ashi | renko (a display transform: "
                                      "signals and indicators still use the real candles)",
                                      autocompletion=complete.choices(["candles", "line", "heikin-ashi", "renko"]))] = "candles",
    brick: Annotated[Optional[float], typer.Option(help="Renko brick size in price units (default: ATR 14)")] = None,
    compare: Annotated[Optional[list[str]], typer.Option("--compare", help="Overlay these symbols as percent change "
                                                         "from the left edge (repeat the option); needs no indicators")] = None,
):
    """Terminal OHLC chart with indicator overlays, volume and algorithm signals.

    ``--type heikin-ashi|renko`` change how the candles are drawn; ``--compare ETH/USDT --compare SOL/USDT`` plots
    percent change of several symbols instead. A ratio chart is a symbol like ``ETH/USDT:BTC/USDT`` or ``KO:PEP``."""
    from tradebot.analysis.charttypes import TYPES, heikin_ashi, percent_change, renko
    from tradebot.cli.charts import render_chart, render_lines

    if kind not in TYPES:
        raise typer.BadParameter(f"type must be one of {', '.join(TYPES)}", param_hint="--type")
    if compare:
        names = [symbol, *compare]
        frame = percent_change({n: load_data(n, tf, None, None, bars)["close"] for n in names})
        if frame.empty:
            raise typer.BadParameter("the symbols have no candles in common")
        print(render_lines({n: frame[n] for n in names}, f"{', '.join(names)} {tf}: change in %"))
        return

    algorithm = make_algo(algo, param, members, entry, exit, fills, sizing, direction) if algo else None
    inds = [parse_indicator(s) for s in indicator]
    if algorithm and not inds:
        inds = algorithm.chart_indicators()
    df = load_data(symbol, tf, None, None, bars + 250)  # extra history warms up indicators
    shown = None
    title = f"{symbol} {tf}" + (f" ({kind})" if kind not in ("candles", "line") else "")
    if kind == "heikin-ashi":
        shown = heikin_ashi(df)
    elif kind == "renko":  # bricks have their own time stamps: no indicators, signals or volume next to them
        try:
            df, inds, algorithm, volume = renko(df.tail(bars), brick), [], None, False
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="--brick") from None
    print(render_chart(df, title, inds, algorithm, line or kind == "line", window=bars, volume=volume, shown=shown))


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
    entry: Entry = None,
    exit: Exit = None,
    fills: Fills = None,
    sizing: Sizing = None,
    direction: Direction = None,
    leverage: Leverage = 1.0,
    maintenance_margin: MaintenanceMargin = 0.005,
    borrow_fee: BorrowFee = 0.0,
    funding: Funding = None,
    fee: Annotated[float, typer.Option(help="Fee per side, fraction")] = 0.001,
    slippage: Slippage = None,
    capital: float = 10_000.0,
    dca_amount: DcaAmount = None,
    plot: Annotated[bool, typer.Option(help="Show chart with signals and equity")] = True,
    trades_csv: Annotated[Optional[Path], typer.Option("--trades-csv", help="Write the trade list to this CSV file")] = None,
    report: Annotated[Optional[Path], typer.Option(help="Write a self-contained HTML report (metrics, equity, "
                                                        "underwater plot, monthly returns, trades)")] = None,
    portfolio: Annotated[bool, typer.Option(
        "--portfolio", help="Trade all the symbols from one shared account (#148) instead of one backtest per "
                            "symbol; compared with buy & hold of the same basket")] = False,
    watchlist: Annotated[list[str], typer.Option(
        "--watchlist", "-w", help="Add the symbols of this watchlist (with --portfolio; 'all' = every list)",
        autocompletion=complete.watchlists)] = [],
    max_positions: Annotated[Optional[int], typer.Option(
        min=1, help="Portfolio: most positions held at once, each slot 1/N of the account (default: one per "
                    "symbol). Implies --portfolio")] = None,
    allocation: Annotated[str, typer.Option(
        help="Portfolio: how the held positions share the capital: equal, signal (by signal strength) or "
             "volatility (inverse volatility)",
        autocompletion=complete.choices(("equal", "same share each"), ("signal", "by signal strength"),
                                        ("volatility", "inverse volatility")))] = "equal",
    rebalance: Annotated[str, typer.Option(
        help="Portfolio: set the positions back to their target weights: none, candle, week, month or quarter",
        autocompletion=complete.choices(("none", "never"), ("candle", "every candle"), ("week", "weekly"),
                                        ("month", "monthly"), ("quarter", "quarterly")))] = "none",
    regimes: Annotated[bool, typer.Option("--regimes", help="Break the result down by market regime (bull / bear / "
                                                            "sideways, high / low volatility)")] = False,
):
    """Backtest an algorithm on historical OHLC data, next to buy & hold and monthly investing.

    ``combo`` builds a strategy from rules: ``tradebot backtest AAPL combo --entry "elliott_wave &
    close>sma(200)" --exit "macd_cross | stop_atr(2)"`` (see ``tradebot combo --help``).

    ``--portfolio`` trades several symbols from one account: ``tradebot backtest AAPL,MSFT,NVDA,KO sma_cross
    --max-positions 2 --allocation volatility --rebalance month``."""
    from tradebot.backtest import backtest as run_backtest, baselines
    from tradebot.cli.charts import render_chart

    algorithm = make_algo(algo, param, members, entry, exit, fills, sizing, direction)
    margin = _margin(leverage, maintenance_margin, borrow_fee, funding)
    if portfolio or max_positions or watchlist:
        if algorithm.allow_short or margin is not None or funding:
            raise typer.BadParameter("the portfolio backtest is long-only without leverage (no --direction, "
                                     "--leverage, --borrow-fee or --funding)")
        _backtest_portfolio(_symbols(symbol, watchlist), algorithm, tf, start, end, bars, fee, capital, slippage,
                            max_positions, allocation, rebalance, plot, trades_csv, report)
        return
    symbols = _symbols(symbol, [])
    if len(symbols) > 1:
        _backtest_many(symbols, algorithm, tf, start, end, bars, fee, capital, dca_amount, slippage, margin, funding)
        return
    df = load_data(symbol, tf, start, end, bars)
    slip = _slippage(slippage, symbol)
    try:
        res = run_backtest(df, algorithm, fee=fee, initial_capital=capital, slippage=slip, margin=margin,
                           funding=_funding_history(funding, symbol, df))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    if res.liquidations is not None and res.liquidations.any():
        when = ", ".join(f"{t:%Y-%m-%d}" for t in res.liquidations.index[res.liquidations.to_numpy()][:5])
        console.print(f"[bold red]liquidated {int(res.liquidations.sum())} time(s)[/]: {when}")
    if plot:
        print(render_chart(df, f"{symbol} {tf} · {algorithm!r}", algorithm.chart_indicators(), algorithm, equity=res.equity))
    bl = baselines(df, res, fee=fee, initial_capital=capital, amount=dca_amount, slippage=slip)
    console.print(f"[bold]{algorithm!r}[/] on {symbol} {tf} ({df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d})")
    console.print(results_table("backtest", bl.columns("strategy", res)))
    console.print(f"final equity [bold]{res.equity.iloc[-1]:,.2f}[/]")
    from tradebot.backtest import report as rep

    stats = rep.report_stats(res.equity, res.trades)
    console.print("  ".join(f"{k.replace('_', ' ')} [bold]{rep.fmt_stat(k, v)}[/]" for k, v in stats.items()))
    by_regime = rep.regime_table(res, df) if regimes or report else None
    if regimes:
        f = rep.fmt_regime(by_regime)
        tbl = Table(title="by market regime")
        tbl.add_column("regime")
        for c in f.columns:
            tbl.add_column(c.replace("_", " "), justify="right")
        for name, row in zip(f.index, f.itertuples(index=False)):
            tbl.add_row(name, *row)
        console.print(tbl)
    if trades_csv:
        console.print(f"trades written to {rep.export_trades_csv(res, trades_csv)}")
    if report:
        console.print(f"report written to {rep.export_html(res, report, f'{algorithm!r} on {symbol} {tf}', by_regime)}")
    warn_not_beaten(bl, "Over this period")


@app.command()
def montecarlo(
    symbol: Symbol,
    algo: Algo,
    tf: TF = "1d",
    start: Annotated[Optional[str], typer.Option(help="ISO date")] = None,
    end: Annotated[Optional[str], typer.Option(help="ISO date")] = None,
    bars: Annotated[int, typer.Option(help="Number of candles (ignored with --start)")] = 1000,
    param: Params = [],
    members: Members = None,
    entry: Entry = None,
    exit: Exit = None,
    fee: Annotated[float, typer.Option(help="Fee per side, fraction")] = 0.001,
    runs: Annotated[int, typer.Option(help="Number of simulated re-orderings")] = 1000,
    method: Annotated[str, typer.Option(help="shuffle (same trades, random order) | bootstrap (draw with "
                                        "replacement) | block (bootstrap in runs of consecutive items)",
                                        autocompletion=complete.choices(["shuffle", "bootstrap", "block"]))] = "bootstrap",
    source: Annotated[str, typer.Option(help="trades (per round trip) | returns (per candle)",
                                        autocompletion=complete.choices(["trades", "returns"]))] = "trades",
    block: Annotated[Optional[int], typer.Option(help="Block length (method block); default sqrt(items)")] = None,
    max_dd: Annotated[float, typer.Option(help="Report the probability of a drawdown above this fraction")] = 0.2,
    seed: Annotated[Optional[int], typer.Option(help="Random seed, for a repeatable result")] = None,
):
    """Monte Carlo robustness test: how much of a backtest is luck in the order of its trades?

    Re-draws the trades (or candle returns) ``--runs`` times and prints 5 / 50 / 95 % bands for the final
    return and the maximum drawdown, plus the probability of a loss and of a drawdown above ``--max-dd``."""
    from tradebot.analysis.montecarlo import METHODS, SOURCES, from_backtest
    from tradebot.backtest import backtest as run_backtest

    if method not in METHODS:
        raise typer.BadParameter(f"method must be one of {', '.join(METHODS)}", param_hint="--method")
    if source not in SOURCES:
        raise typer.BadParameter(f"source must be one of {', '.join(SOURCES)}", param_hint="--source")
    if runs < 1:
        raise typer.BadParameter("runs must be at least 1", param_hint="--runs")
    algorithm = make_algo(algo, param, members, entry, exit)
    df = load_data(symbol, tf, start, end, bars)
    res = run_backtest(df, algorithm, fee=fee)
    try:
        mc = from_backtest(res, runs, source=source, method=method, block_length=block, dd_threshold=max_dd, seed=seed)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    console.print(f"[bold]{algorithm!r}[/] on {symbol} {tf}: {mc.runs} {mc.method} runs over {mc.items} {mc.source}"
                  + (f" in blocks of {mc.block_length}" if mc.block_length else ""))
    t = Table(title="Monte Carlo")
    for col, just in (("metric", "left"), ("backtest", "right"), ("5%", "right"), ("median", "right"), ("95%", "right"),
                      ("probability", "right")):
        t.add_column(col, justify=just)
    for label, actual, band, prob in mc.rows():
        t.add_row(label, *(["–"] * 4 if band is None else [f"{actual:.1%}", *(f"{v:.1%}" for v in band)]),
                  "–" if prob is None else f"{prob:.1%}")
    console.print(t)
    if mc.method == "shuffle":
        console.print("[dim]Shuffling keeps the final return of every run; only the drawdown depends on the order.[/]")



def _slippage(value: str | None, symbol: str | None) -> float:
    """``--slippage`` (basis points, 'auto' or unset) as a fraction."""
    from tradebot.backtest import bps
    from tradebot.backtest.slippage import resolve_bps

    try:
        return bps(resolve_bps(value, symbol))
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--slippage") from None


def _backtest_many(symbols, algorithm, tf, start, end, bars, fee, capital, dca_amount, slippage=None, margin=None,
                   funding=None) -> None:
    """One row per symbol: the strategy's return, Sharpe, XIRR and lead over each baseline."""
    from tradebot.backtest import backtest as run_backtest, baselines
    from tradebot.backtest.metrics import fmt

    t = Table(title=f"{algorithm!r} · {tf}")
    for col in ("symbol", "total return", "sharpe", "XIRR", "vs buy & hold", "vs monthly", "trades"):
        t.add_column(col, justify="left" if col == "symbol" else "right")
    beats = {}
    for s, df in _load_many(symbols, tf, start, end, bars).items():
        slip = _slippage(slippage, s)
        res = run_backtest(df, algorithm, fee=fee, initial_capital=capital, slippage=slip, margin=margin,
                           funding=_funding_history(funding, s, df))
        bl = baselines(df, res, fee=fee, initial_capital=capital, amount=dca_amount, slippage=slip)
        lead = [f"[{'green' if v > 0 else 'red'}]{fmt('vs buy & hold', v)}[/]" for v in bl.excess.values()]
        for name in bl.excess:
            beats[name] = beats.get(name, 0) + (name not in bl.not_beaten)
        t.add_row(s, fmt("total_return", res.metrics["total_return"]), fmt("sharpe", res.metrics["sharpe"]),
                  fmt("xirr", bl.strategy_xirr), *lead, str(len(res.trades)))
    console.print(t)
    n = len(t.rows)
    console.print(" · ".join(f"beats {k} on {v}/{n}" for k, v in beats.items()))


def _backtest_portfolio(symbols, algorithm, tf, start, end, bars, fee, capital, slippage, max_positions, allocation,
                        rebalance, plot=False, trades_csv=None, report=None) -> None:
    """One account trading all ``symbols`` (#148): metrics next to the basket's buy & hold, then per symbol."""
    from tradebot.backtest import report as rep
    from tradebot.backtest.metrics import fmt
    from tradebot.backtest.portfolio import ALLOCATIONS, REBALANCE, portfolio_backtest
    from tradebot.cli.charts import render_lines

    if allocation not in ALLOCATIONS:
        raise typer.BadParameter(f"choose {', '.join(ALLOCATIONS)}", param_hint="--allocation")
    if rebalance not in REBALANCE:
        raise typer.BadParameter(f"choose {', '.join(REBALANCE)}", param_hint="--rebalance")
    frames = _load_many(symbols, tf, start, end, bars)
    slip = {s: _slippage(slippage, s) for s in frames}
    try:
        pf = portfolio_backtest(frames, algorithm, max_positions, allocation, rebalance, fee=fee, slippage=slip,
                                initial_capital=capital)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    res, idx = pf.result, pf.result.equity.index
    if plot:
        print(render_lines({"portfolio": res.equity, "buy & hold basket": pf.benchmark.equity},
                           f"portfolio · {algorithm!r} · {tf}"))
    s = pf.settings
    console.print(f"[bold]{algorithm!r}[/] on {len(frames)} symbols {tf} ({idx[0]:%Y-%m-%d} → {idx[-1]:%Y-%m-%d}), "
                  f"max {s['max_positions']} positions, {allocation} allocation, rebalance {rebalance}")
    console.print(results_table("portfolio backtest", pf.columns()))
    console.print(f"final equity [bold]{res.equity.iloc[-1]:,.2f}[/] (basket {pf.benchmark.equity.iloc[-1]:,.2f})")
    t = Table(title="per symbol")
    for col in ("symbol", "P&L", "time held", "avg weight", "trades"):
        t.add_column(col, justify="left" if col == "symbol" else "right")
    for sym, row in pf.per_symbol().iterrows():
        color = "green" if row["contribution"] > 0 else "red"
        t.add_row(sym, f"[{color}]{row['contribution']:+,.2f}[/]", fmt("exposure", row["time held"]),
                  fmt("exposure", row["avg weight"]), str(int(row["trades"])))
    console.print(t)
    if trades_csv:
        console.print(f"trades written to {rep.export_trades_csv(res, trades_csv)}")
    if report:
        console.print(f"report written to {rep.export_html(res, report, f'{algorithm!r} portfolio · {tf}')}")
    if not pf.excess > 0:
        console.print("[bold red]Over this period, the portfolio does not beat buy & hold of the same basket[/]")


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
    entry: Entry = None,
    exit: Exit = None,
    fills: Fills = None,
    sizing: Sizing = None,
    direction: Direction = None,
    leverage: Leverage = 1.0,
    maintenance_margin: MaintenanceMargin = 0.005,
    borrow_fee: BorrowFee = 0.0,
    funding: Funding = None,
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
    slippage: Slippage = None,
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
        if sizing or direction:
            from tradebot.algorithms.direction import directed
            from tradebot.algorithms.sizing import sized

            algorithm = sized(directed(algorithm, direction), sizing)
        algo = "pool"
    elif pool or min_members != 1 or max_members is not None:
        raise typer.BadParameter("--pool, --min-members and --max-members need the pool: leave out the algorithm")
    else:
        algorithm = make_algo(algo, [], members, entry, exit, fills, sizing, direction)
    if len(symbols) == 1:
        df = load_data(symbols[0], tf, start, end, bars)
        size = f"{len(df)} candles"
    else:
        df = _load_many(symbols, tf, start, end, bars)
        size = f"{len(df)} symbols · {sum(map(len, df.values()))} candles"
    if backend == "auto" and not algorithm.vectorized:
        backend = "numpy"  # loop-based algorithms (e.g. elliott_wave) run on the CPU anyway
    kwargs = dict(population=population, generations=generations, objective=objective, train_fraction=train,
                  backend=select_backend(backend), fee=fee, slippage=_slippage(slippage, symbols[0]), seed=seed, fixed=parse_kv(fix), min_trades=min_trades,
                  dca_amount=dca_amount, aggregate=aggregate, align=align, min_symbols_beating=min_symbols_beating,
                  samples=samples, sample_length=sample_length)
    if funding == "history":
        raise typer.BadParameter("the optimizer takes a constant rate per 8 hours; historical funding is for "
                                 "backtest", param_hint="--funding")
    if (margin := _margin(leverage, maintenance_margin, borrow_fee, funding)) is not None:
        kwargs["margin"] = margin
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
    ctx: typer.Context,
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
    higher_tf: Annotated[Optional[str], typer.Option(
        "--higher-tf", help="Comma-separated higher timeframes, e.g. 1d,1w: the rules may also use every indicator "
        "computed on those candles (only closed ones, no look-ahead)")] = None,
    permutations: Annotated[Optional[int], typer.Option(
        help="Also run the search this many times on shuffled candles, to see whether it finds as much in noise "
        "(default 0, with --from-scratch 5)")] = None,
    folds: Annotated[Optional[int], typer.Option(
        help="Walk-forward: also run the search on this many rolling windows, each scored on the stretch after it "
        "(default 0, with --from-scratch 3)")] = None,
    fee: float = 0.001,
    slippage: Slippage = None,
    dca_amount: DcaAmount = None,
    best_result: Annotated[bool, typer.Option(
        "--best-result", help="Set every setting for the best out-of-sample result, ignoring runtime (much slower "
        "than the defaults): bigger search, shuffled-price runs, walk-forward folds, limits scaled to the data. "
        "Flags you give yourself still win")] = False,
    effort: Annotated[str, typer.Option(
        help="With --best-result or --auto-settings: good (somewhat reasonable run time) | best (ignores the run "
        "time: more search, more shuffled-price runs and folds, and with --auto-settings every candidate setting "
        "gets them too)", autocompletion=complete.choices(
            ("good", "reasonable run time"), ("best", "ignores run time")))] = "best",
    auto_settings: Annotated[bool, typer.Option(
        "--auto-settings", help="Let a search pick the settings for this data first (successive halving over many "
        "settings, ranked by validation score minus the overfitting gap; the newest --test part is kept out of it "
        "and used once by the final run). Takes much longer than --best-result. Flags you give yourself still win")] = False,
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
    higher = _higher_timeframes(higher_tf, tf)
    if from_scratch and higher:
        raise typer.BadParameter("from scratch uses no ready-made indicators", param_hint="--higher-tf")
    symbols = _symbols(symbol, watchlist)
    df = load_data(symbols[0], tf, start, end, bars) if len(symbols) == 1 else _load_many(symbols, tf, start, end, bars)
    if best_result or auto_settings:
        from tradebot.evolve import auto_configure, best_result_settings

        if effort not in ("good", "best"):
            raise typer.BadParameter("good or best", param_hint="--effort")
        candles = min(len(f) for f in df.values()) if isinstance(df, dict) else len(df)
        if auto_settings:
            fixed = dict(fee=fee, slippage=_slippage(slippage, symbols[0]), align=align, dca_amount=dca_amount, state=state)
            if names:
                fixed["indicators"] = names
            if higher:
                fixed["timeframes"] = higher
            from rich.progress import Progress as _Progress

            with _Progress(console=console) as bar:
                task = bar.add_task("auto-configuring", total=None)
                try:
                    auto = auto_configure(df, from_scratch=from_scratch, objective=objective, level=effort,
                                          seed=seed, **fixed,
                                          on_trial=lambda i, n, label: bar.update(
                                              task, total=n, completed=i, description=f"auto-configuring: {label}"))
                except ValueError as e:
                    raise typer.BadParameter(str(e)) from None
            for line in auto.report():
                console.print(f"[dim]{line}[/]")
            preset = auto.settings
        else:
            preset = best_result_settings(candles, len(symbols), from_scratch=from_scratch, level=effort)
        given = {n for n in ctx.params if getattr(ctx.get_parameter_source(n), "name", "") == "COMMANDLINE"}
        population, generations, max_depth, max_size, islands, validation, test, segments, min_trades, parsimony, \
            aggregate, permutations, folds = (
                v if n in given else preset[n] for n, v in (
                    ("population", population), ("generations", generations), ("max_depth", max_depth),
                    ("max_size", max_size), ("islands", islands), ("validation", validation), ("test", test),
                    ("segments", segments), ("min_trades", min_trades), ("parsimony", parsimony),
                    ("aggregate", aggregate), ("permutations", permutations), ("folds", folds)))
        extra = {k: preset[k] for k in ("tournament", "crossover", "mutation", "elite", "hall_of_fame")}
        if "state" not in given:
            state = preset["state"]
        console.print(f"[dim]{'auto-configured' if auto_settings else 'best result'} ({effort}): population {population}, {generations} generations, depth {max_depth}, "
                      f"size {max_size}, {permutations} shuffled runs, {folds} folds, {segments} segments, "
                      f"min trades {min_trades}; this takes a long time[/]")
    else:
        extra = {}
    kwargs = dict(**extra, population=population, generations=generations, max_depth=max_depth, max_size=max_size,
                  objective=objective, aggregate=aggregate, align=align, validation=validation, test=test,
                  segments=segments, min_trades=min_trades, parsimony=parsimony, state=state, indicators=names,
                  timeframes=higher, fee=fee, slippage=_slippage(slippage, symbols[0]), dca_amount=dca_amount, seed=seed, from_scratch=from_scratch, islands=islands)
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
    from tradebot.algorithms.direction import Directed
    from tradebot.algorithms.sizing import Sized

    base, wrappers = algorithm, ""  # sizing (#147) and direction (#149) wrappers are flags
    while isinstance(base, (Sized, Directed)):
        wrappers += f" --sizing {base.model}" if isinstance(base, Sized) else f" --direction {base.direction}"
        base = base.inner
    if type(base).__name__ == "Combo":  # the generic combo: its rules aren't saved anywhere
        import shlex

        members = f" --entry {shlex.quote(base.entry)}" + (f" --exit {shlex.quote(base.exit)}" if base.exit else "")
    if getattr(base, "fills", "close") != "close":
        members += f" --fills {base.fills}"
    members += wrappers
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
    entry: Entry = None,
    exit: Exit = None,
    fills: Fills = None,
    sizing: Sizing = None,
    live: Annotated[bool, typer.Option(help="Actually place orders (default: dry run)")] = False,
    sandbox: Annotated[bool, typer.Option(help="Use the exchange testnet (ccxt brokers)")] = False,
    once: Annotated[bool, typer.Option(help="Evaluate a single candle and exit")] = False,
    max_daily_loss: Annotated[float, typer.Option(help="Halt for the day after losing this much (quote currency)")] = 0.0,
    max_drawdown: Annotated[float, typer.Option(help="Halt when equity falls this many % of the stake below its peak")] = 0.0,
    max_exposure: Annotated[float, typer.Option(help="Reject entries that would push the position above this value")] = 0.0,
    max_order: Annotated[float, typer.Option(help="Reject orders above this value")] = 0.0,
    price_band: Annotated[float, typer.Option(help="Reject entries when the exchange price is this many % off the last close")] = 0.0,
    cooldown_losses: Annotated[int, typer.Option(help="Pause entries after this many losing trades in a row")] = 0,
    cooldown_minutes: Annotated[float, typer.Option(help="Length of that pause")] = 60.0,
    close_on_halt: Annotated[bool, typer.Option(help="Sell the open position when a limit halts the bot")] = False,
    paper: Annotated[bool, typer.Option(help="Trade a simulated account that survives restarts (see `tradebot paper`)")] = False,
    journal: Annotated[bool, typer.Option(help="Store every order in the trade journal (`tradebot journal`)")] = True,
):
    """Run an algorithm live against a broker (dry run unless --live or --paper). Limits are off at 0; `bot kill` stops it."""
    from tradebot.journal import Journal
    from tradebot.live import RiskLimits, TradingBot

    if paper:
        broker = "paper"
    confirm_live_risk(live)

    kwargs = {"sandbox": True} if sandbox else {}
    b = create_broker(broker, **kwargs)
    if live and not b.can_trade:
        raise typer.BadParameter(f"{broker} is read-only")
    limits = RiskLimits(max_daily_loss or None, max_drawdown or None, max_exposure or None, max_order or None,
                        price_band or None, cooldown_losses, cooldown_minutes, close_on_halt)
    bot = TradingBot(b, make_algo(algo, param, members, entry, exit, fills, sizing), symbol, tf, stake,
                     dry_run=not (live or broker == "paper"), risk=limits, journal=Journal() if journal else None, on_risk=lambda e: console.print(f"[red]risk guard {e.kind}:[/] {e.message}"),
                     on_event=lambda e: console.print(
                         f"[dim]{e.time:%Y-%m-%d %H:%M}[/] {symbol} target={e.target:.2f} "
                         f"price={e.price:.6g} → [bold]{e.action}[/] {e.amount:.6g}"
                         + (" [yellow](dry run)[/]" if e.dry_run else "")
                         + (f" order {e.order.id} {e.order.status.value}" if e.order else "")))
    mode = "[red]LIVE[/]" if live else "[cyan]paper[/]" if broker == "paper" else "[yellow]dry run[/]"
    console.print(f"bot {algo} on {broker}:{symbol} {tf}, stake {stake} · {mode} · {b.execution_mode.value}")
    try:
        bot.run(max_steps=1 if once else None)
    except KeyboardInterrupt:
        console.print("stopped")


@bot_app.command("kill")
def bot_kill():
    """Kill switch: every running `bot run` stops trading at its next check (#154)."""
    from tradebot.live.risk import kill_file, set_kill

    set_kill(True)
    console.print(f"[red]kill switch on[/] ({kill_file()}); `tradebot bot resume` turns it off")


@bot_app.command("resume")
def bot_resume():
    """Turn the kill switch off. A bot that already halted needs a restart."""
    from tradebot.live.risk import set_kill

    set_kill(False)
    console.print("kill switch off")



paper_app = typer.Typer(help="The simulated paper trading account.", no_args_is_help=True)
app.add_typer(paper_app, name="paper")


@paper_app.command("show")
def paper_show(orders: Annotated[int, typer.Option(help="How many recent orders to list")] = 10):
    """Cash, positions, P&L and the latest orders of the paper account."""
    from tradebot.brokers.paper import PaperBroker

    b = PaperBroker()
    s = b.summary()
    cur = s["currency"]
    console.print(f"[bold]paper account[/] ({b.path}) · fee {b.fee:.2%}")
    console.print(f"cash {s['cash']:,.2f} {cur} · positions {s['positions_value']:,.2f} · equity {s['equity']:,.2f} "
                  f"(start {s['starting_balance']:,.2f})")
    console.print(f"P&L {s['pnl']:+,.2f} {cur} ({s['pnl_pct']:+.2%}) · realised {s['realised_pnl']:+,.2f} · "
                  f"fees {s['fees']:,.2f} · {s['orders']} orders")
    for a, p in b.positions.items():
        console.print(f"  {a}: {p['quantity']:.8g} @ avg {p['avg_cost']:.6g}")
    for o in b.orders[-orders:] if orders > 0 else []:
        console.print(f"[dim]{o['time'][:16]}[/] {o['side']} {o['amount']:.6g} {o['symbol']} @ {o['price']:.6g}"
                      + (f" P&L {o['profit']:+.2f}" if o["profit"] is not None else ""))


@paper_app.command("reset")
def paper_reset(balance: Annotated[float, typer.Option(help="New starting balance (default: keep the current one)")] = 0.0,
                yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask")] = False):
    """Throw away positions and order history and start again with a fresh balance."""
    from tradebot.brokers.paper import PaperBroker

    b = PaperBroker()
    if not yes:
        typer.confirm(f"Reset the paper account ({len(b.orders)} orders, equity {b.summary()['equity']:,.2f})?", abort=True)
    b.reset(balance or None)
    console.print(f"paper account reset to {b.cash:,.2f} {b.currency}")


journal_app = typer.Typer(help="The trade journal: every order the bots placed (#159).", no_args_is_help=True)
app.add_typer(journal_app, name="journal")
Mode = Annotated[str, typer.Option(help="live, paper, dry (dry runs) or real (live + paper)")]


@journal_app.command("show")
def journal_show(mode: Mode = "real", symbol: Annotated[Optional[str], typer.Option(help="Only this symbol")] = None,
                 tag: Annotated[Optional[str], typer.Option(help="Only fills with this tag")] = None,
                 limit: Annotated[int, typer.Option(help="How many recent fills to list (0: none)")] = 20):
    """Realised P&L, fees and the latest fills with their ids, notes and tags."""
    from tradebot.journal import Journal, yearly_summary

    j = Journal()
    s = j.summary(mode)
    wr = "–" if s["win_rate"] is None else f"{s['win_rate']:.0%}"
    console.print(f"[bold]journal[/] ({j.path}) · {s['fills']} fills · {s['lots']} closed lots · win rate {wr}")
    console.print(f"realised {s['realised']:+,.2f} · fees {s['fees']:,.2f} · {s['open']} open piece(s)"
                  " (unrealised P&L needs prices: see the Journal tab)")
    for y in yearly_summary(j.lots(mode)):
        console.print(f"  {y['year']}: net {y['net']:+,.2f} (gains {y['gains']:,.2f}, losses {y['losses']:,.2f}, "
                      f"short-term {y['short_term']:+,.2f}, long-term {y['long_term']:+,.2f}) in {y['lots']} lots")
    for f in (j.trades(mode, symbol, tag)[-limit:] if limit > 0 else []):
        console.print(f"[dim]#{f['id']} {f['time'][:16]}[/] {f['mode']} {f['side']} {f['amount']:.6g} {f['symbol']} @ "
                      f"{f['price']:.6g} fee {f['fee']:.4g}" + (f" P&L {f['pnl']:+.2f}" if f["pnl"] is not None else "")
                      + (f" [{escape(f['tags'])}]" if f["tags"] else "") + (f" {escape(f['note'])}" if f["note"] else ""))


@journal_app.command("note")
def journal_note(fill_id: Annotated[int, typer.Argument(help="Fill id, as shown by `journal show`")],
                 note: Annotated[Optional[str], typer.Option(help="Note text")] = None,
                 tags: Annotated[Optional[str], typer.Option(help="Comma-separated tags (replaces the old ones)")] = None):
    """Set the note and/or tags of a fill."""
    from tradebot.journal import Journal

    if note is None and tags is None:
        raise typer.BadParameter("give --note and/or --tags")
    if not Journal().annotate(fill_id, note, tags):
        raise typer.BadParameter(f"no fill #{fill_id}")
    console.print(f"fill #{fill_id} updated")


@journal_app.command("export")
def journal_export(path: Annotated[Path, typer.Argument(help="CSV file to write")],
                   lots: Annotated[bool, typer.Option(help="Write the FIFO tax lots instead of the fills")] = False,
                   year: Annotated[Optional[int], typer.Option(help="With --lots: only lots closed in this year")] = None,
                   mode: Mode = "real"):
    """Export the fills (with notes, tags, realised P&L) or, with --lots, the FIFO tax lots per year as CSV."""
    from tradebot.journal import Journal

    j = Journal()
    n = j.export_lots(path, mode, year) if lots else j.export_trades(path, mode)
    console.print(f"{n} {'lots' if lots else 'fills'} written to {path}")


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
    journal: Annotated[bool, typer.Option(help="Store every order in the trade journal (`tradebot journal`)")] = True,
):
    """Trade the gap of two assets live: long one and short the other (dry run unless --live).

    The short leg is a plain sell, so the broker needs margin or futures for it: give the legs as symbols it
    can short. With --live the open pair is saved (`bot pair-state` shows it) and picked up after a restart;
    a trade that didn't finish makes the next start refuse until the account has been checked."""
    from tradebot.journal import Journal
    from tradebot.live import PairBot, PairStateError, pair_state_path

    legs, symbol = (leg_a, leg_b), f"{leg_a} / {leg_b}"
    confirm_live_risk(live)
    b = create_broker(broker, **({"sandbox": True} if sandbox else {}))
    if live and not b.can_trade:
        raise typer.BadParameter(f"{broker} is read-only")
    try:
        bot = PairBot(b, legs, tf, stake, window, entry, exit, stop, method, dry_run=not live,
                      state_path=pair_state_path(broker, legs, sandbox) if live else None,
                      journal=Journal() if journal else None,
                      on_event=lambda e: console.print(
                          f"[dim]{e.time:%Y-%m-%d %H:%M}[/] {e.pair} z={e.z:+.2f} β={e.beta:.2f} → [bold]{e.action}[/]"
                          + (f" {e.amount_a:.6g} {legs[0]} / {e.amount_b:.6g} {legs[1]}" if e.action != "hold" else "")
                          + (" [yellow](dry run)[/]" if e.dry_run else "")
                          + (f" [yellow]({e.note})[/]" if e.note else "")))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    except PairStateError as e:
        console.print(f"[bold red]not starting:[/] {e}")
        raise typer.Exit(1) from None
    mode = "[red]LIVE[/]" if live else "[yellow]dry run[/]"
    console.print(f"pair bot {symbol} on {broker} {tf}, stake {stake} · {mode} · {b.execution_mode.value}")
    if bot.side:
        console.print(f"resumed the open pair: {'long' if bot.side > 0 else 'short'} {bot.amounts[0]:.8g} {legs[0]} / "
                      f"{bot.amounts[1]:.8g} {legs[1]} (since {bot.opened})")
    try:
        bot.run(max_steps=1 if once else None)
    except KeyboardInterrupt:
        console.print("stopped")
    except PairStateError as e:
        console.print(f"[bold red]halted:[/] {e}")
        raise typer.Exit(1) from None


@bot_app.command("pair-state")
def bot_pair_state(
    leg_a: Annotated[str, typer.Argument(help="First leg, as given to pair-run")],
    leg_b: Annotated[str, typer.Argument(help="Second leg")],
    broker: Broker = "binance",
    sandbox: Annotated[bool, typer.Option(help="The testnet state (pair-run --sandbox)")] = False,
    keep: Annotated[bool, typer.Option(help="Drop an unfinished trade, keep the last saved pair (it matches the account)")] = False,
    forget: Annotated[bool, typer.Option(help="Mark the pair flat (you closed both legs at the broker)")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask")] = False,
):
    """Show what `bot pair-run --live` remembers for a pair; --keep / --forget clear an unfinished trade (#122)."""
    from datetime import datetime, timezone

    from tradebot.live.pairs import (
        STATE_VERSION,
        PairStateError,
        describe_pending,
        pair_state_path,
        read_pair_state,
        write_json_atomic,
    )

    if keep and forget:
        raise typer.BadParameter("use --keep or --forget, not both")
    where = f"{broker}{' testnet' if sandbox else ''}"
    path = pair_state_path(broker, (leg_a, leg_b), sandbox)
    try:
        s = read_pair_state(path)
    except PairStateError as e:
        console.print(f"[bold red]{e}[/]")
        if not forget:
            raise typer.Exit(1) from None
        s = None
    if s is None and not forget:
        console.print(f"no saved pair for {leg_a} / {leg_b} on {where} ({path})")
        return
    if s is not None:
        side = {1.0: "long", -1.0: "short"}.get(s["side"], "flat")
        held = f" {s['amounts'][0]:.8g} {leg_a} / {s['amounts'][1]:.8g} {leg_b} since {s.get('opened')}" if s["side"] else ""
        console.print(f"[bold]{leg_a} / {leg_b}[/] on {where} ({path})")
        console.print(f"{side}{held}" + (" · closing" if s["closing"] else "") + f" · updated {s.get('updated')}")
        if s.get("pending"):
            console.print(f"[bold red]unfinished trade:[/] {describe_pending(s)}")
    if not (keep or forget):
        return
    if keep and s is not None and not s.get("pending"):
        console.print("nothing to clear")
        return
    if not yes:
        typer.confirm("Mark the pair flat?" if forget else "Keep the saved pair and drop the unfinished trade?", abort=True)
    if forget:
        s = {"version": STATE_VERSION, "legs": [leg_a, leg_b], "side": 0.0, "amounts": [0.0, 0.0], "closing": False,
             "opened": None}
    s["pending"], s["updated"] = None, datetime.now(timezone.utc).isoformat()
    write_json_atomic(path, s)
    console.print("pair marked flat" if forget else "unfinished trade dropped; the saved pair stays")


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


@app.command("portfolio-analytics")
def portfolio_analytics(
    broker: Annotated[list[str], typer.Option(help="Broker(s) to aggregate",
                                              autocompletion=complete.brokers)] = ["scalable"],
    benchmark: Annotated[str, typer.Option(help="Benchmark symbol for the comparison, e.g. SPY, URTH or BTC/USDT",
                                           autocompletion=complete.symbols)] = "SPY",
    target: Annotated[Optional[str], typer.Option(
        help='Target allocation, e.g. "AAPL=40,MSFT=60,BTC/USDT=0": prints what to buy and sell')] = None,
    tolerance: Annotated[float, typer.Option(help="Leave holdings within this many percentage points of target")] = 0.0,
    mode: Annotated[str, typer.Option(help="Journal fills to build the history from: real, live, paper or dry")] = "real",
    days: Annotated[int, typer.Option(help="Price history to load for correlation and risk")] = 365,
    plot: Annotated[bool, typer.Option("--plot/--no-plot", help="Draw the return curves in the terminal")] = True,
):
    """Portfolio over time (time- and money-weighted return vs a benchmark, from the journal), correlation matrix,
    risk shares and rebalancing suggestions toward a target allocation."""
    from tradebot.data import MarketDataEngine
    from tradebot.journal import Journal
    from tradebot.portfolio import build_portfolio
    from tradebot.portfolio_analytics import analyse, parse_targets

    engine = MarketDataEngine()
    positions_ = [p for name in broker for p in create_broker(name, data=engine).get_positions()]
    targets = parse_targets(target) if target else None
    journal = Journal()
    fills = journal.fills(mode)
    journal.close()
    first = min((pd.Timestamp(f["time"]) for f in fills), default=None)
    first = first.tz_localize("UTC") if first is not None and first.tzinfo is None else first
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    start = min(since, first - pd.Timedelta(days=5)) if first is not None else since
    with console.status("pricing holdings and loading history ..."):
        pf = build_portfolio(positions_, engine.latest_price)
        out = analyse(lambda s: engine.get_ohlc(s, "1d", start=start)["close"], pf, fills, benchmark, targets,
                      tolerance / 100)
    _print_analytics(out, plot, benchmark)


def _print_analytics(out: dict, plot: bool, benchmark: str) -> None:
    pct = lambda v: "–" if v is None or v != v else f"{v * 100:+.2f}%"  # noqa: E731
    perf = out["performance"]
    if perf and perf["summary"]:
        s = perf["summary"]
        if plot:
            lines = {"portfolio (TWR)": perf["twr"] * 100}
            if len(perf["bench"]):
                lines[benchmark] = perf["bench"] * 100
            console.print(render_lines(lines, "return since first fill, %"))
        console.print(f"value [bold]{s['value']:,.2f}[/] · paid in {s['invested']:,.2f} · profit {s['profit']:,.2f}\n"
                      f"time-weighted {pct(s['twr'])} · money-weighted {pct(s['mwr'])} (annualised {pct(s['mwr_annual'])}) · "
                      f"{benchmark} {pct(s['benchmark'])} · vs benchmark {pct(s['excess'])}")
    else:
        console.print("[yellow]No journal fills with price data: the performance history needs orders recorded "
                      "in the journal (tradebot journal).[/]")
    if out["correlation"] is not None:
        corr = out["correlation"]
        t = Table(title="correlation of daily returns")
        t.add_column("")
        for c in corr.columns:
            t.add_column(c, justify="right")
        for sym, row in corr.iterrows():
            t.add_row(sym, *[f"{v:.2f}" for v in row])
        console.print(t)
        t = Table(title="risk")
        for c in ("symbol", "weight", "volatility", "share of risk"):
            t.add_column(c, justify="left" if c == "symbol" else "right")
        for sym, r in out["risk"].sort_values("risk_share", ascending=False).iterrows():
            t.add_row(sym, f"{r.weight * 100:.1f}%", f"{r.volatility * 100:.1f}%", f"{r.risk_share * 100:.1f}%")
        console.print(t)
    reb = out["rebalance"]
    if reb is not None:
        t = Table(title="rebalancing to target")
        for c in ("symbol", "value", "weight", "target", "action", "amount", "quantity"):
            t.add_column(c, justify="left" if c in ("symbol", "action") else "right")
        for _, r in reb.iterrows():
            colour = {"buy": "green", "sell": "red"}.get(r.action, "dim")
            t.add_row(r.symbol, f"{r.value:,.2f}", f"{r.weight * 100:.1f}%", f"{r.target * 100:.1f}%",
                      f"[{colour}]{r.action}[/]", f"{r.trade_value:+,.2f}", f"{r.quantity:+,.6g}")
        console.print(t)


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
    indicator_tf: Annotated[Optional[str], typer.Option(
        "--indicator-tf", autocompletion=complete.timeframes,
        help="Run the indicator on this longer timeframe, its last closed candle (e.g. -t 1h --indicator-tf 1d)")] = None,
    where: Annotated[list[str], typer.Option("--where", help='e.g. --where "fast > slow" --where "rsi < 30"')] = [],
    alert: Annotated[list[str], typer.Option("--alert", help="Alert that must be on; !NAME: off. e.g. --alert cross_up")] = [],
    watchlist: Annotated[Optional[str], typer.Option("--watchlist", "-w", help=WATCHLIST_HELP)] = None,
    index: Annotated[Optional[str], typer.Option("--index", help="Scan a market index instead: sp500, nasdaq100, "
                                                 "dow30, dax40, ftse100, crypto_top100 (weekly cached list)",
                                                 autocompletion=complete.indices)] = None,
    notify: Annotated[bool, typer.Option("--notify", help="Send the matches to the notification channels "
                                         "(event kind scanner)")] = False,
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
    if index:
        if symbols or watchlist:
            raise typer.BadParameter("use either symbols, --watchlist or --index", param_hint="--index")
        symbols = _index_symbols(index)
    symbols = symbols or _watchlist(watchlist)
    try:
        with console.status("scanning ..."):
            result = scanner.scan(symbols, indicator, params, timeframe, filters, indicator_tf=indicator_tf)
    except (ValueError, KeyError, scanner.ScanBusy) as exc:
        raise typer.BadParameter(str(exc.args[0] if exc.args else exc)) from None
    t = Table(title=f"scan: {indicator}{f' ({indicator_tf})' if indicator_tf else ''} {timeframe}")
    for c in result.table.columns:
        t.add_column(c, justify="left" if c == "symbol" else "right")
    for _, row in result.table.iterrows():
        t.add_row(*[str(row[c]) if isinstance(row[c], (bool, np.bool_)) else _fmt_cell(c, row[c]) for c in result.table.columns])
    console.print(t)
    console.print(result.summary())
    if notify:
        from tradebot.notify import notify as send_notification

        names = ", ".join(result.table["symbol"].astype(str)) if len(result.table) else "no matches"
        send_notification("scanner", f"Scan {indicator} {timeframe}: {result.matched} match", names, wait=True)
    for sym, why in list(result.skipped.items())[:5]:
        console.print(f"[dim]skipped {sym}: {why}[/]")


WATCHLIST_HELP = "Without symbols: only this watchlist's (default: all watchlists)"


def _index_symbols(key: str) -> list[str]:
    from tradebot import universes
    from tradebot.data.cache import DiskCache

    if key not in universes.INDICES:
        raise typer.BadParameter(f"unknown index {key!r}; one of: {', '.join(universes.INDICES)}", param_hint="--index")
    cache = DiskCache.from_config()
    return universes.index_symbols(key, cache.symbols_dir if cache else None)


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


@combo_app.command("save")
def combo_save(
    name: Annotated[str, typer.Argument(help="Name to save under (lower-case letters, digits and _)")],
    entry: Annotated[str, typer.Option(help="Entry rule, e.g. \"elliott_wave & close>sma(200)\"")],
    exit: Annotated[Optional[str], typer.Option(help="Exit rule, e.g. \"macd_cross | stop_atr(2)\"")] = None,
    description: Annotated[str, typer.Option(help="Shown in tradebot list and the GUI")] = "",
    param: Params = [],
    fills: Fills = "close",
):
    """Save a combo by name; it then works like any strategy: list, backtest, optimize, bot, chart, GUI.

    Rules: strategy names (in the exit rule: their exit), indicator conditions (close>sma(200), rsi(14)<30,
    elliott.state==3), & | ! and brackets, vote(a, b, c) >= 0.6, weighted(a:0.5, b:0.3) >= 0.6, and in the
    exit rule the risk exits stop_pct(5) stop_atr(2) take_profit(10) take_profit_atr(3) trail_pct(5)
    trail_atr(3) max_hold(20); take-profits in steps scale out: take_profit(5, 0.5) | take_profit(10). In the
    entry rule a limit entry: & limit_pct(1) or & limit_atr(0.5) buys below the close on the next candle.
    ``--fills intrabar`` fills stops, take-profits and limit entries inside the candle against its high/low.
    ``-p`` sets saved parameter values (e.g. tuned ones from ``optimize``)."""
    from tradebot.algorithms.combo import combos_path, save_combo

    try:
        save_combo(name, entry, exit or "", description, parse_kv(param), fills=fills or "close")
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    console.print(f"saved combo [bold]{name}[/] in {combos_path()}")


@combo_app.command("list")
def combo_list():
    """The saved combos and their rules."""
    from tradebot.algorithms.combo import combos_path, load_saved

    saved = load_saved()
    if not saved:
        console.print(f"no saved combos yet ({combos_path()}); save one with: tradebot combo save NAME --entry ...")
        return
    t = Table(title=f"combos · {combos_path()}")
    for col in ("name", "entry", "exit", "fills", "parameters"):
        t.add_column(col, style="bold cyan" if col == "name" else None)
    for name, c in saved.items():
        params = ", ".join(f"{k}={v:g}" for k, v in (c.get("params") or {}).items())
        t.add_row(name, c["entry"], c.get("exit") or "[dim]when the entry stops holding[/]", c.get("fills", "close"),
                  params)
    console.print(t)


@combo_app.command("delete")
def combo_delete(name: Annotated[str, typer.Argument(help="The saved combo to delete")]):
    """Delete a saved combo."""
    from tradebot.algorithms.combo import delete_combo

    if not delete_combo(name):
        raise typer.BadParameter(f"no saved combo {name!r}")
    console.print(f"deleted combo {name}")


@notify_app.command("test")
def notify_test():
    """Send a test message to every enabled channel and report which worked."""
    from tradebot import notify as notifications

    chans = notifications.channels()
    if not chans:
        console.print("no channels: add [notify.<name>] tables to config.toml (see docs/settings.md)")
        raise typer.Exit(1)
    failed = 0
    for name, error in notifications.test_all().items():
        failed += bool(error)
        console.print(f"[red]✗[/] {name}: {escape(error)}" if error else f"[green]✓[/] {name}")
    if failed:
        raise typer.Exit(1)


# -- background alerts (#156) ------------------------------------------------------------------------
alert_app = typer.Typer(help="Price and indicator alerts and scheduled scans, delivered through the notification "
                             "channels while `tradebot watch` (or the app) runs.", no_args_is_help=True)
app.add_typer(alert_app, name="alert")
Every = Annotated[str, typer.Option("--every", help="Minutes between checks, or 'close' = once per finished candle")]
Repeat = Annotated[bool, typer.Option("--repeat", help="Re-arm after the condition was false again "
                                      "(default: fire once, then switch off)")]


def _store_alert(alert):
    from tradebot import alerts

    try:
        alerts.add(alert)
    except (ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc.args[0] if exc.args else exc)) from None
    console.print(f"added {alert.id}: {escape(alert.describe())} "
                  f"(every {alert.every}{'' if alert.every == 'close' else ' min'})")


def _scan_params(items: list[str]) -> dict:
    params = {}
    for item in items:
        key, _, value = item.partition("=")
        try:
            params[key] = float(value)
        except ValueError:
            raise typer.BadParameter(f"{item!r}: use name=number", param_hint="--param") from None
    return params


def _alert_filters(where: list[str], alert: list[str]) -> list[dict]:
    return [parse_where(w) for w in where] + [{"alert": a.lstrip("!"), "state": not a.startswith("!")} for a in alert]


@alert_app.command("price")
def alert_price(symbol: Symbol,
                condition: Annotated[str, typer.Argument(help="e.g. '>=100000' or '<=90000'")],
                timeframe: Annotated[str, typer.Option("--tf", help="Candle the price is read from",
                                                       autocompletion=complete.timeframes)] = "1m",
                every: Every = "1", repeat: Repeat = False):
    """Tell me when SYMBOL reaches a price."""
    from tradebot import alerts

    try:
        op, value = alerts.parse_price(condition)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="CONDITION") from None
    _store_alert(alerts.Alert("price", [symbol], timeframe, op=op, value=value, every=every, repeat=repeat))


@alert_app.command("indicator")
def alert_indicator(
    symbol: Symbol,
    indicator: Annotated[str, typer.Option("--indicator", "-i", autocompletion=complete.scannable)],
    param: Annotated[list[str], typer.Option("--param", "-p", help="e.g. -p period=14")] = [],
    where: Annotated[list[str], typer.Option("--where", help='e.g. --where "rsi < 30"')] = [],
    alert: Annotated[list[str], typer.Option("--alert", help="Indicator alert that must be on; !NAME: off")] = [],
    timeframe: Annotated[str, typer.Option("--tf", autocompletion=complete.timeframes)] = "1d",
    every: Every = "close", repeat: Repeat = False,
):
    """Tell me when an indicator's values or alerts on SYMBOL match (all conditions)."""
    from tradebot import alerts

    _store_alert(alerts.Alert("indicator", [symbol], timeframe, indicator=indicator, params=_scan_params(param),
                              filters=_alert_filters(where, alert), every=every, repeat=repeat))


@alert_app.command("scan")
def alert_scan(
    symbols: Annotated[Optional[list[str]], typer.Argument(help="Defaults to the watchlists",
                                                           autocompletion=complete.symbols)] = None,
    indicator: Annotated[str, typer.Option("--indicator", "-i", autocompletion=complete.scannable)] = "ma_cross",
    param: Annotated[list[str], typer.Option("--param", "-p")] = [],
    where: Annotated[list[str], typer.Option("--where")] = [],
    alert: Annotated[list[str], typer.Option("--alert")] = [],
    timeframe: Annotated[str, typer.Option("--tf", autocompletion=complete.timeframes)] = "1d",
    watchlist: Annotated[Optional[str], typer.Option("--watchlist", "-w", help=WATCHLIST_HELP)] = None,
    index: Annotated[Optional[str], typer.Option("--index", autocompletion=complete.indices)] = None,
    every: Every = "close",
):
    """Run a scan on a schedule and report only the matches that are new since the last run."""
    from tradebot import alerts

    if index:
        if symbols or watchlist:
            raise typer.BadParameter("use either symbols, --watchlist or --index", param_hint="--index")
        symbols = _index_symbols(index)
    symbols = symbols or _watchlist(watchlist)
    _store_alert(alerts.Alert("scan", symbols, timeframe, indicator=indicator, params=_scan_params(param),
                              filters=_alert_filters(where, alert), every=every))


@alert_app.command("list")
def alert_list():
    """The stored alerts and their state."""
    from tradebot import alerts

    items = alerts.load()
    if not items:
        console.print("no alerts: add one with `tradebot alert price|indicator|scan`")
        return
    t = Table(title="alerts")
    for c in ("id", "kind", "condition", "every", "state"):
        t.add_column(c)
    for a in items:
        state = "off" if not a.enabled else "armed" if a.armed else "fired (waiting)"
        t.add_row(a.id, a.kind, escape(a.describe()), str(a.every), state)
    console.print(t)


@alert_app.command("rm")
def alert_rm(alert_id: Annotated[str, typer.Argument(help="Id from `tradebot alert list`")]):
    """Delete an alert."""
    from tradebot import alerts

    if not alerts.remove(alert_id):
        raise typer.BadParameter(f"no alert {alert_id!r}")
    console.print(f"removed {alert_id}")


@app.command()
def watch(poll: Annotated[float, typer.Option(help="Seconds between passes over the alerts")] = 30.0,
          once: Annotated[bool, typer.Option("--once", help="One pass, then exit (for cron)")] = False):
    """Run the stored alerts and scheduled scans in the foreground (headless; Ctrl-C stops).

    Don't run it next to the app's own alert timer: both would fire the same alert.
    """
    from tradebot import alerts

    if once:
        for line in alerts.run_due():
            console.print(escape(line))
        return
    console.print(f"watching {len(alerts.load())} alerts every {poll:g} s (Ctrl-C to stop)")
    try:
        alerts.watch(poll, log_fn=lambda s: console.print(escape(s)))
    except KeyboardInterrupt:
        console.print("stopped")


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
