# tradebot documentation

tradebot is an extensible algorithmic trading platform for crypto and stocks: multi-source market data, charts with
indicators, a vectorized backtester, a (GPU-accelerated) optimizer, a strategy evolver, live and paper trading
bots, a desktop app (PyQt6) and a terminal CLI. It is licensed under the GNU GPL v3.0 or later.

> **Read this first.** tradebot is independent software, not financial advice, and provided without any warranty.
> Backtests don't predict future returns. See [Disclaimer and licence](legal.md) before you trade real money.

## Where to start

* New here: [Getting started](getting-started.md), then open the app with `tradebot gui`.
* Prefer the terminal: [Command line](cli.md).
* Want to know whether a strategy is any good: [Metrics](metrics.md) and
  [How the optimizers work](optimizers.md).
* Writing your own indicator or strategy: [Extending](extending.md) and
  [Indicators as algorithms](indicators-as-algorithms.md).

## Contents

**Getting started**

* [Getting started](getting-started.md): install, the binary, GPU, first run, the data cache

**Desktop app** (`tradebot gui`), one page per sidebar entry

* [Chart](chart.md): candles, indicators, loadouts, drawings, watchlists and news in the sidebar
* [Screener](screener.md)
* [Scanner](scanner.md): one indicator over a watchlist or an index, plus the script editor
* [Backtest lab](backtest-lab.md): strategies, optimizer, walk-forward, meta-search, baselines, Monte Carlo, reports
* [Evolve](evolve.md): strategies written from scratch by genetic programming
* [Pairs](pairs.md)
* [Portfolio](portfolio.md)
* [Basket](portfolio-backtest.md): one account trading a strategy on several symbols (portfolio backtest)
* [Journal](journal.md): every order the bots placed, realised P&L, tax lots
* [News](news.md)
* [Settings](settings.md)

**Command line**

* [Command line](cli.md): every command with examples, shell completion
* [CLI reference](cli-reference.md): every option, generated from the code

**Extending**

* [Extending](extending.md): plugins (indicators, algorithms, brokers, notification channels), the script editor
* [Indicators as algorithms](indicators-as-algorithms.md): turn an indicator into something that trades

**Concepts**

* [Strategies](strategies.md): the built-in strategies, `vote`, `pool`, the combo builder, Elliott waves, halving cycle
* [Metrics](metrics.md): what every number in the results means
* [How the optimizers work](optimizers.md): GA, PSO, walk-forward, meta-search, pool, slippage, baselines
* [Data providers, timeframes and the cache](data.md)
* [Brokers and live trading](live-trading.md): brokers, bots, paper trading, risk guards, notifications
* [Pairs trading](pairs-trading.md)
* [Market regimes and position sizing](regimes-and-sizing.md)
* [Pattern detection](patterns.md): candlestick and chart patterns, support and resistance
* [Crypto market data](market-data.md): funding rate, open interest, Fear & Greed, on-chain metrics, order book depth
* [Multi-timeframe indicators and strategies](multi-timeframe.md): `tf=` in rules, on the chart, in the Scanner and in evolve
* [Shorts and leverage](shorts-and-leverage.md): long/short strategies, leverage, liquidation, borrow fees, funding
* [Alerts](alerts.md): price, indicator and scheduled-scan alerts

**Reference**

* [Reference](reference.md): `config.toml` keys, `settings.json`, environment variables, file locations

**Development and legal**

* [Development](development.md): tests, CI, branches, releases
* [Disclaimer and licence](legal.md)

## Conventions

* Commands are shown for an installed `tradebot`; from a source checkout write `uv run tradebot ...`.
* Symbols are `AAPL`, `SAP.DE`, `BTC/USDT` (crypto pairs have a slash) and pairs `KO:PEP`.
* Timeframes: `1m 5m 15m 30m 1h 2h 4h 1d 1w 1mo`. Not every command accepts all of them; `--help` lists them.
* `#123` refers to an issue or pull request in the
  [GitHub repository](https://github.com/daMarxrider/trading_bot_3).
