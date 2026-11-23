# Command line

One binary is the CLI and the desktop app. Every command has `--help`; the complete list of options is in the
[CLI reference](cli-reference.md), generated from the code. `tradebot --version` prints the version and the
[disclaimer](legal.md); `-v` / `--verbose` turns on logging.

Charts in the terminal are rendered with plotext, so they work in bash, zsh and fish.

## Discover

```sh
tradebot list algorithms        # strategies with their parameters and ranges
tradebot list indicators
tradebot list brokers
tradebot device                 # which compute backend the optimizer will use (CUDA, ROCm, NumPy)
tradebot gui                    # the desktop application
```

## Charts and data

```sh
tradebot chart BTC/USDT --tf 4h --bars 200 -i bbands -i rsi --algo rsi_threshold
tradebot chart AAPL -i sma:period=50 --line --no-volume
tradebot news AAPL NVDA BTC/USDT
tradebot news --watchlist Crypto           # without symbols: one watchlist (default: all of them)
tradebot news BTC/USDT -s coindesk -s decrypt   # only some sources (`--list-sources` shows them)
tradebot market BTC/USDT --depth           # funding rate, open interest, Fear & Greed, MVRV, order book (see Crypto market data)
tradebot cache info                        # cache folder, size, number of cached series
tradebot cache clear [--ohlc|--symbols]
```

`tradebot market` and the market data indicators are described in [Crypto market data](market-data.md).
`--bars N` returns exactly N candles; `--start`/`--end` select a date range. See
[Data providers, timeframes and the cache](data.md).

## Backtest

```sh
tradebot backtest AAPL macd_cross --bars 2000 -p fast=8 -p slow=21
tradebot backtest AAPL,MSFT,NVDA macd_cross -p fast=8       # one row per symbol
tradebot backtest AAPL vote --members rsi_threshold,sma_cross
tradebot backtest AAPL combo --entry "close > sma(200) & rsi(14) < 35" --exit "stop_pct(5) | rsi(14) > 55"
tradebot backtest AAPL,MSFT,NVDA,KO,JNJ sma_cross --max-positions 3 --allocation volatility --rebalance month   # one shared account, see portfolio-backtest.md
tradebot backtest BTC/USDT sma_cross --sizing vol -p size.target_vol=0.2   # position sizing
tradebot backtest BTC/USDT macd_cross --direction long_short --leverage 3 --funding history   # shorts and leverage, see shorts-and-leverage.md
tradebot backtest AAPL combo --entry "rsi_threshold & regime.bull" --regimes   # result per market regime
tradebot backtest AAPL macd_cross --slippage 5 --fee 0.001
tradebot backtest AAPL macd_cross --report aapl.html --trades-csv aapl.csv --no-plot
tradebot montecarlo AAPL macd_cross --runs 2000 --method block --seed 1
```

`backtest` prints the metrics next to **buy & hold** and **monthly investing** ([Metrics](metrics.md#baselines)).
`--slippage 5` charges 5 basis points per side on top of the fee; `--slippage auto` uses a guess for the asset class;
the default comes from `[backtest] slippage_bps` in `config.toml`. `--fills intrabar` applies to combos
([Strategies](strategies.md#the-combo-builder)).

## Optimize and evolve

```sh
tradebot optimize BTC/USDT rsi_threshold --population 1024 --generations 40 --objective calmar
tradebot optimize BTC/USDT rsi_threshold --method pso          # particle swarm instead of the genetic algorithm
tradebot optimize BTC/USDT rsi_threshold --method both         # run GA and PSO, then compare them
tradebot optimize BTC/USDT                                     # also pick which strategies/indicators to use
tradebot optimize BTC/USDT --pool rsi_threshold,macd_cross,elliott_wave,sma --max-members 3
tradebot optimize AAPL,MSFT,NVDA,SAP.DE macd_cross --aggregate median   # one parameter set for several symbols
tradebot optimize -w Crypto sma_cross                          # every symbol of a watchlist (-w all: of all of them)
tradebot optimize -w all rsi_threshold --min-symbols-beating 5 --samples 4 --sample-length 300
tradebot optimize AAPL macd_cross --walk-forward 5 --holdout 250   # re-optimize in 5 folds, then a holdout
tradebot optimize -w Crypto sma_cross --meta 20                 # also tune the optimizer's own settings
tradebot optimize AAPL rsi_threshold --fix period=14            # keep a parameter fixed
tradebot evolve AAPL,MSFT,NVDA --generations 60 --save my_rule  # evolve a whole strategy from scratch
```

What the options do: [How the optimizers work](optimizers.md) and [Evolve](evolve.md). `optimize` tunes on the first
70 % of the data (`--train`) and reports the rest out-of-sample; candidates with fewer than `--min-trades` round trips
are rejected. `--backend auto|gpu|cuda|rocm|cpu|numpy` picks the compute backend.

## Alerts

```sh
tradebot alert price BTC/USDT ">=100000" --every 5 --repeat
tradebot alert indicator AAPL -i rsi --where "value < 30" --tf 1d
tradebot alert scan -i ma_cross --alert cross_up --index sp500
tradebot alert list
tradebot alert rm ID
tradebot watch                      # run the alerts until Ctrl-C (--once for cron)
```

See [Alerts](alerts.md). Portfolio backtests are described in [Basket](portfolio-backtest.md), `--sizing` and `--regimes` in
[Market regimes and position sizing](regimes-and-sizing.md).

## Combos

```sh
tradebot combo save trend_dip --entry "close > sma(200) & rsi(14) < 35" --exit "rsi(14) > 55 | stop_pct(5)"
tradebot combo list
tradebot combo delete trend_dip
```

A saved combo works like any strategy name everywhere ([Strategies](strategies.md#the-combo-builder)).

## Screen and scan

```sh
tradebot screen -f "rsi < 35" -f "close > sma200"
tradebot scan --alert cross_up --notify         # the Scanner page's engine: one indicator over many symbols
tradebot scan -i rsi --where "value < 30" --index sp500 -t 1d
```

See [Screener](screener.md) and [Scanner](scanner.md).

## Pairs and halving

```sh
tradebot pair KO:PEP                                    # gap z-score, hedge ratio, half-life, cointegration
tradebot pair-backtest KO:PEP --window 60 --entry 2
tradebot pair-backtest KO:PEP --method kalman
tradebot pair-scan KO PEP PG MDLZ
tradebot pair-scan KO PEP PG MDLZ --same-sector --correction bonferroni
tradebot pair-scan $(cat sp500.txt) --backend gpu       # big universes: all tests batched on a GPU
tradebot halving --live                                 # Bitcoin halvings and the forecast of the next
```

See [Pairs trading](pairs-trading.md) and [Strategies](strategies.md#bitcoin-halving-cycle).

## Bots, paper trading, brokers

```sh
tradebot bot run ETH/USDT sma_cross --broker binance --tf 1h --stake 50          # dry run
tradebot bot run ETH/USDT sma_cross --broker binance --tf 1h --stake 50 --live   # real orders
tradebot bot run ETH/USDT sma_cross --tf 1h --stake 50 --paper                   # simulated account that survives restarts
tradebot bot run ETH/USDT sma_cross --stake 50 --max-daily-loss 20 --max-drawdown 10 --cooldown-losses 3
tradebot bot pair-run BTC/USDT:USDT ETH/USDT:USDT --once
tradebot bot pair-state BTC/USDT:USDT ETH/USDT:USDT   # the open pair a --live pair bot saved (--keep / --forget)
tradebot bot kill                   # every running bot stops at its next check
tradebot bot resume
tradebot paper show                 # paper account: cash, positions, P&L, latest orders
tradebot paper reset --balance 5000
tradebot positions --broker kucoin
tradebot portfolio --broker scalable --broker binance
tradebot portfolio-analytics --benchmark SPY --target "AAPL=60,MSFT=40"   # return vs benchmark, correlation, risk, rebalancing
tradebot journal show               # trade journal: realised P&L per year, fees, latest fills
tradebot journal note 12 --note "news-driven" --tags swing,eth
tradebot journal export lots_2026.csv --lots --year 2026   # FIFO tax lots
tradebot notify test                # send a test message to every notification channel
```

The first `bot run ... --live` shows the risk notice and asks you to accept it. Bots record every order in the [journal](journal.md) (`--no-journal` switches it off). Everything about brokers, the risk
guards and notifications: [Brokers and live trading](live-trading.md).

## Tab completion

Bash, fish, zsh and PowerShell. Run this once, *from the shell you want it in*; it detects the shell and installs the
completion script (bash: `~/.bash_completions/`, fish: `~/.config/fish/completions/tradebot.fish`). Open a new shell
afterwards.

```sh
tradebot --install-completion
tradebot --show-completion      # print the script instead, to install it by hand
```

TAB completes commands and options, algorithm and indicator names (plugins included), the `-p`/`--fix` parameters of
the chosen algorithm with defaults and ranges, `-i name:param=`, `--members`, brokers, timeframes and fixed choices.
Symbols complete by ticker, case-insensitively: your watchlists first, then crypto pairs (`sol` becomes `SOL/USDT`) and
stocks/ETFs from Yahoo (`nvd` becomes `NVDA`). Shells match suggestions against what you typed, so unlike the GUI you
can't find a ticker by company name here. The `tradebot` command must be on your `PATH` (an activated venv,
`uv tool install`, or the standalone binary). Lookups are cached in the OS cache folder, so after the first TAB a
completion takes about half a second.
