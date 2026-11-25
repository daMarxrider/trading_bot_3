# CLI reference

Every command and option of `tradebot`, generated from the command definitions by `python scripts/check_docs.py --write`; CI fails when this page is out of date. For examples see [Command line](cli.md); `tradebot COMMAND --help` prints the same.

## `tradebot`

Algorithmic trading platform: charts, backtests, optimization, live bots.

| Option | Default | Description |
|---|---|---|
| `-v`, `--verbose` |  |  |
| `--version` |  | Show the version and the disclaimer |
| `--install-completion` |  | Install completion for the current shell. |
| `--show-completion` |  | Show completion for the current shell, to copy it or customize the installation. |

## `tradebot list`

List registered algorithms, indicators or brokers (built-ins and plugins).

Arguments: `KIND` (optional)

## `tradebot chart`

Terminal OHLC chart with indicator overlays, volume and algorithm signals. ``--type heikin-ashi\|renko`` change how the candles are drawn; ``--compare ETH/USDT --compare SOL/USDT`` plots percent change of several symbols instead. A ratio chart is a symbol like ``ETH/USDT:BTC/USDT`` or ``KO:PEP``.

Arguments: `SYMBOL`

| Option | Default | Description |
|---|---|---|
| `--tf` | `1d` | Timeframe: 1m 5m 15m 30m 1h 4h 1d 1w |
| `--bars` | `150` | Number of candles |
| `--indicator`, `-i` |  | e.g. -i sma:period=50 -i rsi; tf= on a higher timeframe: -i sma:period=200,tf=1d |
| `--algo` |  | Overlay this algorithm's buy/sell signals |
| `--param`, `-p` |  | Algorithm parameter, e.g. -p lower=30 |
| `--members` |  | Comma-separated members for the 'vote' ensemble, or for the 'pool' (strategies and indicators) |
| `--entry` |  | Combo entry rule (with the 'combo' strategy), e.g. "elliott_wave & close>sma(200)" or "weighted(rsi_threshold:0.5, macd_cross:0.5) >= 0.6" |
| `--exit` |  | Combo exit rule, e.g. "macd_cross \| stop_atr(2)" (default: when the entry rule stops holding) |
| `--fills` |  | Combo fills: close (stops, take-profits and limit entries checked on the close) or intrabar (filled inside the candle against its high/low, at the open on a gap; stop first when both are hit) |
| `--sizing` |  | Position sizing on top of the strategy (#147): fixed (fraction of the capital), vol (volatility target), risk (fixed risk per trade, the ATR stop distance decides the size) or kelly (fractional Kelly). Tune it with -p size.<name>=value, e.g. -p size.target_vol=0.15 |
| `--direction` |  | Trade direction (#149): long (default), short (short on the strategy's sell signal) or long_short (long on its buy signal, short on its sell signal). Shorts are backtested with liquidation and --borrow-fee / --funding; the live bot does not open shorts |
| `--line`, `--no-line` |  | Line chart instead of candles |
| `--volume`, `--no-volume` | `True` | Volume bars under the candles (when the data has volume) |
| `--type` | `candles` | candles \| line \| heikin-ashi \| renko (a display transform: signals and indicators still use the real candles) |
| `--brick` |  | Renko brick size in price units (default: ATR 14) |
| `--compare` |  | Overlay these symbols as percent change from the left edge (repeat the option); needs no indicators |

## `tradebot backtest`

Backtest an algorithm on historical OHLC data, next to buy & hold and monthly investing. ``combo`` builds a strategy from rules: ``tradebot backtest AAPL combo --entry "elliott_wave & close>sma(200)" --exit "macd_cross \| stop_atr(2)"`` (see ``tradebot combo --help``). ``--portfolio`` trades several symbols from one account: ``tradebot backtest AAPL,MSFT,NVDA,KO sma_cross --max-positions 2 --allocation volatility --rebalance month``.

Arguments: `SYMBOL`, `ALGO`

| Option | Default | Description |
|---|---|---|
| `--tf` | `1d` | Timeframe: 1m 5m 15m 30m 1h 4h 1d 1w |
| `--start` |  | ISO date |
| `--end` |  | ISO date |
| `--bars` | `1000` | Number of candles (ignored with --start) |
| `--param`, `-p` |  | Algorithm parameter, e.g. -p lower=30 |
| `--members` |  | Comma-separated members for the 'vote' ensemble, or for the 'pool' (strategies and indicators) |
| `--entry` |  | Combo entry rule (with the 'combo' strategy), e.g. "elliott_wave & close>sma(200)" or "weighted(rsi_threshold:0.5, macd_cross:0.5) >= 0.6" |
| `--exit` |  | Combo exit rule, e.g. "macd_cross \| stop_atr(2)" (default: when the entry rule stops holding) |
| `--fills` |  | Combo fills: close (stops, take-profits and limit entries checked on the close) or intrabar (filled inside the candle against its high/low, at the open on a gap; stop first when both are hit) |
| `--sizing` |  | Position sizing on top of the strategy (#147): fixed (fraction of the capital), vol (volatility target), risk (fixed risk per trade, the ATR stop distance decides the size) or kelly (fractional Kelly). Tune it with -p size.<name>=value, e.g. -p size.target_vol=0.15 |
| `--direction` |  | Trade direction (#149): long (default), short (short on the strategy's sell signal) or long_short (long on its buy signal, short on its sell signal). Shorts are backtested with liquidation and --borrow-fee / --funding; the live bot does not open shorts |
| `--leverage` | `1.0` | Leverage (#149): each position is this many times its margin. Positions are liquidated (the margin is lost) when the price reaches the liquidation price |
| `--maintenance-margin` | `0.005` | Maintenance margin rate, fraction of the notional (0.005 = 0.5%); liquidation below it |
| `--borrow-fee` |  | Yearly borrow fee of a short, fraction (stocks ~0.01-0.05, perpetuals 0) |
| `--funding` |  | Funding of a perpetual future: a rate per 8 hours (0.0001 = 0.01%, longs pay shorts when positive) or 'history' for the exchange's historical rates (crypto) |
| `--fee` | `0.001` | Fee per side, fraction |
| `--slippage` |  | Slippage per side in basis points (5 = 0.05%), charged on top of the fee; 'auto' = suggested for the asset class (crypto 5, stock 2). Default: [backtest] slippage_bps in config.toml, else 0 |
| `--no-entry-before-earnings` |  | Open no new position in the DAYS days before a company's earnings date (positions already open are kept). Default: [events] earnings_blackout_days in config.toml, else 0 = off |
| `--capital` | `10000.0` |  |
| `--dca-amount` |  | Amount the monthly-investing baseline invests each month (default: capital / months) |
| `--plot`, `--no-plot` | `True` | Show chart with signals and equity |
| `--trades-csv` |  | Write the trade list to this CSV file |
| `--report` |  | Write a self-contained HTML report (metrics, equity, underwater plot, monthly returns, trades) |
| `--portfolio` |  | Trade all the symbols from one shared account (#148) instead of one backtest per symbol; compared with buy & hold of the same basket |
| `--watchlist`, `-w` |  | Add the symbols of this watchlist (with --portfolio; 'all' = every list) |
| `--max-positions` |  | Portfolio: most positions held at once, each slot 1/N of the account (default: one per symbol). Implies --portfolio |
| `--allocation` | `equal` | Portfolio: how the held positions share the capital: equal, signal (by signal strength) or volatility (inverse volatility) |
| `--rebalance` | `none` | Portfolio: set the positions back to their target weights: none, candle, week, month or quarter |
| `--regimes` |  | Break the result down by market regime (bull / bear / sideways, high / low volatility) |

## `tradebot montecarlo`

Monte Carlo robustness test: how much of a backtest is luck in the order of its trades? Re-draws the trades (or candle returns) ``--runs`` times and prints 5 / 50 / 95 % bands for the final return and the maximum drawdown, plus the probability of a loss and of a drawdown above ``--max-dd``.

Arguments: `SYMBOL`, `ALGO`

| Option | Default | Description |
|---|---|---|
| `--tf` | `1d` | Timeframe: 1m 5m 15m 30m 1h 4h 1d 1w |
| `--start` |  | ISO date |
| `--end` |  | ISO date |
| `--bars` | `1000` | Number of candles (ignored with --start) |
| `--param`, `-p` |  | Algorithm parameter, e.g. -p lower=30 |
| `--members` |  | Comma-separated members for the 'vote' ensemble, or for the 'pool' (strategies and indicators) |
| `--entry` |  | Combo entry rule (with the 'combo' strategy), e.g. "elliott_wave & close>sma(200)" or "weighted(rsi_threshold:0.5, macd_cross:0.5) >= 0.6" |
| `--exit` |  | Combo exit rule, e.g. "macd_cross \| stop_atr(2)" (default: when the entry rule stops holding) |
| `--fee` | `0.001` | Fee per side, fraction |
| `--runs` | `1000` | Number of simulated re-orderings |
| `--method` | `bootstrap` | shuffle (same trades, random order) \| bootstrap (draw with replacement) \| block (bootstrap in runs of consecutive items) |
| `--source` | `trades` | trades (per round trip) \| returns (per candle) |
| `--block` |  | Block length (method block); default sqrt(items) |
| `--max-dd` | `0.2` | Report the probability of a drawdown above this fraction |
| `--seed` |  | Random seed, for a repeatable result |

## `tradebot optimize`

Optimize algorithm parameters with a genetic algorithm and/or particle swarm (GPU accelerated when available) and report out-of-sample results. ``--method both`` also measures how much the two methods disagree - large disagreement means a fragile optimum. Without an algorithm, the optimizer also picks *which* strategies vote and which indicators filter the entries, from all of them or from ``--pool``. With several symbols (comma-separated, or ``--watchlist``) one parameter set is scored on every symbol and the scores are combined (``--aggregate``): settings that only work on one chart lose. ``--meta N`` tunes the optimizer's own settings: N combinations, each ranked by a walk-forward (``--walk-forward`` folds, default 5), the winner scored once on a holdout nothing else has seen.

Arguments: `SYMBOL` (optional), `ALGO` (optional)

| Option | Default | Description |
|---|---|---|
| `--tf` | `1d` | Timeframe: 1m 5m 15m 30m 1h 4h 1d 1w |
| `--start` |  |  |
| `--end` |  |  |
| `--bars` | `2000` | Number of candles (ignored with --start) |
| `--members` |  | Comma-separated members for the 'vote' ensemble, or for the 'pool' (strategies and indicators) |
| `--entry` |  | Combo entry rule (with the 'combo' strategy), e.g. "elliott_wave & close>sma(200)" or "weighted(rsi_threshold:0.5, macd_cross:0.5) >= 0.6" |
| `--exit` |  | Combo exit rule, e.g. "macd_cross \| stop_atr(2)" (default: when the entry rule stops holding) |
| `--fills` |  | Combo fills: close (stops, take-profits and limit entries checked on the close) or intrabar (filled inside the candle against its high/low, at the open on a gap; stop first when both are hit) |
| `--sizing` |  | Position sizing on top of the strategy (#147): fixed (fraction of the capital), vol (volatility target), risk (fixed risk per trade, the ATR stop distance decides the size) or kelly (fractional Kelly). Tune it with -p size.<name>=value, e.g. -p size.target_vol=0.15 |
| `--direction` |  | Trade direction (#149): long (default), short (short on the strategy's sell signal) or long_short (long on its buy signal, short on its sell signal). Shorts are backtested with liquidation and --borrow-fee / --funding; the live bot does not open shorts |
| `--leverage` | `1.0` | Leverage (#149): each position is this many times its margin. Positions are liquidated (the margin is lost) when the price reaches the liquidation price |
| `--maintenance-margin` | `0.005` | Maintenance margin rate, fraction of the notional (0.005 = 0.5%); liquidation below it |
| `--borrow-fee` |  | Yearly borrow fee of a short, fraction (stocks ~0.01-0.05, perpetuals 0) |
| `--funding` |  | Funding of a perpetual future: a rate per 8 hours (0.0001 = 0.01%, longs pay shorts when positive) or 'history' for the exchange's historical rates (crypto) |
| `--pool` |  | Comma-separated strategies and indicators the optimizer picks from (default: all) |
| `--min-members` | `1` | Use at least this many pool members |
| `--max-members` |  | Use at most this many pool members |
| `--fix` |  | Keep a parameter fixed, e.g. --fix period=14 |
| `--method` | `ga` | ga \| pso \| both (run both and compare them) |
| `--population` | `256` | Population (GA) / swarm size (PSO) |
| `--generations` | `30` | Generations (GA) / iterations (PSO) |
| `--objective` | `sharpe` | sharpe \| return \| calmar |
| `--train` | `0.7` | Fraction of data used for training |
| `--backend` | `auto` | auto \| gpu \| cuda \| rocm \| cpu \| numpy |
| `--fee` | `0.001` |  |
| `--slippage` |  | Slippage per side in basis points (5 = 0.05%), charged on top of the fee; 'auto' = suggested for the asset class (crypto 5, stock 2). Default: [backtest] slippage_bps in config.toml, else 0 |
| `--min-trades` | `4` | Reject parameter sets with fewer round trips |
| `--dca-amount` |  | Amount the monthly-investing baseline invests each month (default: capital / months) |
| `--seed` |  |  |
| `--watchlist`, `-w` |  | Also optimize on this watchlist's symbols (repeatable; 'all' = every watchlist) |
| `--aggregate` | `mean` | With several symbols: how their scores combine into one (mean \| median \| min \| mean-std) |
| `--align`, `--no-align` |  | With several symbols: cut all of them to their common date range |
| `--min-symbols-beating` |  | Reject parameter sets that beat buy & hold (in-sample, on the objective) on fewer symbols (0 = off) |
| `--samples` |  | Score each generation on this many random windows per symbol instead of all in-sample candles; new windows every generation, batched in one go on the GPU (0 = off) |
| `--sample-length` | `250` | Candles per random window (with --samples) |
| `--walk-forward` |  | Walk-forward validation: re-optimize in this many folds, each scored on the next unseen stretch (0 = off: one split by --train) |
| `--wf-train` |  | Walk-forward: training candles per fold (default: from --train and the data) |
| `--wf-test` |  | Walk-forward: test candles per fold (default: the folds fill the data) |
| `--anchored`, `--no-anchored` |  | Walk-forward: every training window starts at the first candle and grows, instead of rolling |
| `--holdout` |  | Walk-forward: keep the last N candles out of every fold and score them once at the end (with --meta: 0 = one fold's test length) |
| `--meta` |  | Meta-search: try this many combinations of the optimizer's own settings (random windows, their length, min-symbols-beating), rank them by walk-forward and score the winner once on the holdout (0 = off; Ctrl+C stops early) |
| `--meta-time` |  | Meta-search: stop starting new combinations after this many minutes |
| `--meta-compute`, `--no-meta-compute` |  | Meta-search: also tune population and generations |

## `tradebot evolve`

Evolve a whole strategy from scratch with genetic programming: entry and exit rules built from prices, indicators, arithmetic and logic, with nothing written by hand. The data is split by date: rules evolve on the training part (scored in --segments stretches on every symbol), the best ones are compared on the validation part, and the winner is scored once on the newest --test part, next to buy & hold and monthly investing. --save writes it as a plugin.

Arguments: `SYMBOL` (optional)

| Option | Default | Description |
|---|---|---|
| `--tf` | `1d` | Timeframe: 1m 5m 15m 30m 1h 4h 1d 1w |
| `--start` |  |  |
| `--end` |  |  |
| `--bars` | `3000` | Number of candles (ignored with --start) |
| `--watchlist`, `-w` |  | Also evolve on this watchlist's symbols (repeatable; 'all' = every watchlist) |
| `--population` | `300` | Programs per generation |
| `--generations` | `40` |  |
| `--max-depth` |  | Deepest nesting of a rule (default 6, with --from-scratch 12) |
| `--max-size` |  | Most building blocks (nodes) per rule (default 40, with --from-scratch 100) |
| `--from-scratch` |  | No ready-made indicators: build rules from candles and low-level maths only, so the evolver can invent RSI, Bollinger & co. itself. Turns on the module library, deduplication, the deflated Sharpe ratio, shuffled-price runs and walk-forward folds |
| `--islands` |  | Split the population into this many islands that evolve apart and swap their best now and then (default 1, with --from-scratch 4) |
| `--objective` | `sharpe` | sharpe \| return \| calmar |
| `--aggregate` | `mean` | How the scores of all symbols and segments combine into one (mean \| median \| min \| mean-std) |
| `--align`, `--no-align` |  | With several symbols: cut all of them to their common date range |
| `--validation` | `0.2` | Fraction of the data (after training) that picks the result |
| `--test` | `0.2` | Fraction of the data (the newest) kept for the final, unseen score |
| `--segments` | `3` | The training part is scored in this many consecutive stretches; a rule has to work in all of them |
| `--min-trades` | `4` | Reject programs with fewer round trips per symbol in training |
| `--parsimony` | `0.002` | Fitness penalty per building block: shorter rules win ties |
| `--state`, `--no-state` | `True` | Let exit rules use the open trade (bars in trade, stop loss, take profit, trailing stop) |
| `--indicators` |  | Comma-separated indicators the rules may use (default: all, plugins included) |
| `--higher-tf` |  | Comma-separated higher timeframes, e.g. 1d,1w: the rules may also use every indicator computed on those candles (only closed ones, no look-ahead) |
| `--market-data` |  | Crypto symbols: let the rules also use funding rate, open interest, Fear & Greed and on-chain MVRV / active addresses (fetched and lined up without look-ahead; see `tradebot market`). Naming those indicators in --indicators does the same |
| `--permutations` |  | Also run the search this many times on shuffled candles, to see whether it finds as much in noise (default 0, with --from-scratch 5) |
| `--folds` |  | Walk-forward: also run the search on this many rolling windows, each scored on the stretch after it (default 0, with --from-scratch 3) |
| `--fee` | `0.001` |  |
| `--slippage` |  | Slippage per side in basis points (5 = 0.05%), charged on top of the fee; 'auto' = suggested for the asset class (crypto 5, stock 2). Default: [backtest] slippage_bps in config.toml, else 0 |
| `--dca-amount` |  | Amount the monthly-investing baseline invests each month (default: capital / months) |
| `--best-result` |  | Set every setting for the best out-of-sample result, ignoring runtime (much slower than the defaults): bigger search, shuffled-price runs, walk-forward folds, limits scaled to the data. Flags you give yourself still win |
| `--effort` | `best` | With --best-result or --auto-settings: good (somewhat reasonable run time) \| best (ignores the run time: more search, more shuffled-price runs and folds, and with --auto-settings every candidate setting gets them too) |
| `--auto-settings` |  | Let a search pick the settings for this data first (successive halving over many settings, ranked by validation score minus the overfitting gap; the newest --test part is kept out of it and used once by the final run). Takes much longer than --best-result. Flags you give yourself still win |
| `--seed` |  | Random seed (printed when left out, to repeat a run) |
| `--top` | `5` | Show this many of the best programs with their validation score |
| `--save` |  | Save the result as a plugin algorithm with this name (then usable in backtest, chart, lab and bot) |
| `--plugin-dir` |  | Where --save writes (default: the first plugin folder) |
| `--force`, `--no-force` |  | Let --save overwrite an existing file |

## `tradebot positions`

Show balances / open positions at a broker.

| Option | Default | Description |
|---|---|---|
| `--broker` | `binance` | Registered broker name |

## `tradebot portfolio`

Portfolio value, allocation and dividend metrics (current yield, annual yield, yield on cost).

| Option | Default | Description |
|---|---|---|
| `--broker` | `['scalable']` | Broker(s) to aggregate |

## `tradebot portfolio-analytics`

Portfolio over time (time- and money-weighted return vs a benchmark, from the journal), correlation matrix, risk shares and rebalancing suggestions toward a target allocation.

| Option | Default | Description |
|---|---|---|
| `--broker` | `['scalable']` | Broker(s) to aggregate |
| `--benchmark` | `SPY` | Benchmark symbol for the comparison, e.g. SPY, URTH or BTC/USDT |
| `--target` |  | Target allocation, e.g. "AAPL=40,MSFT=60,BTC/USDT=0": prints what to buy and sell |
| `--tolerance` |  | Leave holdings within this many percentage points of target |
| `--mode` | `real` | Journal fills to build the history from: real, live, paper or dry |
| `--days` | `365` | Price history to load for correlation and risk |
| `--plot`, `--no-plot` | `True` | Draw the return curves in the terminal |

## `tradebot screen`

Screen symbols by technicals, volume, market cap and fundamentals.

Arguments: `SYMBOLS` (optional)

| Option | Default | Description |
|---|---|---|
| `--filter`, `-f` |  | e.g. -f "rsi < 30" -f "close > sma200" |
| `--sort` | `change_1d` |  |
| `--watchlist`, `-w` |  | Without symbols: only this watchlist's (default: all watchlists) |
| `--fundamentals` |  | Add the fundamentals columns (pe, forward_pe, ps, profit_margin, operating_margin, revenue_growth, earnings_growth, dividend_yield, days_to_earnings); also added when a filter uses one, e.g. -f "pe < 15" -f "dividend_yield > 0.03" |

## `tradebot scan`

Scan symbols with one indicator, filtered on its values and alerts (the Scanner tab's engine).

Arguments: `SYMBOLS` (optional)

| Option | Default | Description |
|---|---|---|
| `--indicator`, `-i` | `ma_cross` | Any scannable indicator |
| `--param`, `-p` |  | e.g. -p fast=9 -p slow=21 |
| `--timeframe`, `-t` | `1d` | 1m 5m 15m 30m 1h 2h 4h 1d 1w 1mo |
| `--indicator-tf` |  | Run the indicator on this longer timeframe, its last closed candle (e.g. -t 1h --indicator-tf 1d) |
| `--where` |  | e.g. --where "fast > slow" --where "rsi < 30" |
| `--alert` |  | Alert that must be on; !NAME: off. e.g. --alert cross_up |
| `--watchlist`, `-w` |  | Without symbols: only this watchlist's (default: all watchlists) |
| `--index` |  | Scan a market index instead: sp500, nasdaq100, dow30, dax40, ftse100, crypto_top100 (weekly cached list) |
| `--notify` |  | Send the matches to the notification channels (event kind scanner) |

## `tradebot news`

Latest headlines for symbols (defaults to the watchlists), merged from several RSS feeds and Yahoo.

Arguments: `SYMBOLS` (optional)

| Option | Default | Description |
|---|---|---|
| `--limit` | `25` |  |
| `--watchlist`, `-w` |  | Without symbols: only this watchlist's (default: all watchlists) |
| `--source`, `-s` |  | Only these news sources (repeat the option); default: the ones enabled in the settings. See `tradebot news --list-sources` |
| `--list-sources` |  | Show the available news sources and exit |

## `tradebot market`

Crypto market data beyond candles: funding rate, open interest, Fear & Greed, on-chain MVRV and active addresses, and with --depth the order book (#166). All from free sources that need no key; what can't be fetched (offline, rate limited, no perpetual for the symbol) is named instead. Use the same data as indicators (``funding``, ``open_interest``, ``oi_change``, ``fear_greed``, ``mvrv``, ``active_addresses``) in ``chart``, ``backtest`` rules and ``evolve --market-data``.

Arguments: `SYMBOL`

| Option | Default | Description |
|---|---|---|
| `--tf` | `1d` | Candle timeframe the open interest is bucketed for |
| `--depth` |  | Also show the order book: how much can be traded within 0.5 / 1 / 2 / 5 % of the mid price, and the imbalance |
| `--limit` | `500` | Order book levels per side (with --depth) |

## `tradebot fundamentals`

Valuation, margins, growth and dividend yield of stocks, with their earnings, split and dividend dates. Free data from Yahoo Finance, cached for a day. Crypto pairs have none. Not investment advice.

Arguments: `SYMBOLS` (optional)

| Option | Default | Description |
|---|---|---|
| `--watchlist`, `-w` |  | Without symbols: only this watchlist's (default: all watchlists) |
| `--events`, `--no-events` | `True` | List the earnings, split and ex-dividend dates |
| `--refresh` |  | Ignore the cache and ask again |

## `tradebot pair`

How far two assets that move together are apart now: z-score, hedge ratio, half-life. The pair itself is also a symbol (``chart``, ``backtest``, ... accept ``KO:PEP``): the candles are the ratio of the first to the second, so everything that works on one symbol works on the gap.

Arguments: `SYMBOL`

| Option | Default | Description |
|---|---|---|
| `--tf` | `1d` | Timeframe |
| `--bars` | `500` | Candles to analyse |

## `tradebot pair-backtest`

Backtest trading the gap of two assets: long A and short B (or the reverse) at the same time.

Arguments: `SYMBOL`

| Option | Default | Description |
|---|---|---|
| `--tf` | `1d` | Timeframe |
| `--bars` | `1500` | Candles |
| `--window` | `60` | Candles for the rolling hedge ratio and z-score |
| `--entry` | `2.0` | Open when the gap is this many σ wide |
| `--exit` | `0.5` | Close when it is back inside this many σ |
| `--stop` | `4.0` | Give up beyond this many σ |
| `--fee` | `0.001` | Fee per change of either leg |
| `--borrow` | `0.02` | Yearly cost of the short leg (stocks ~0.02, crypto 0) |
| `--method` | `rolling` | Hedge ratio: rolling (OLS over --window) or kalman (adaptive filter) |

## `tradebot pair-scan`

Find pairs that move together: picked on the first part of the history, confirmed on the rest.

Arguments: `SYMBOLS`

| Option | Default | Description |
|---|---|---|
| `--tf` | `1d` | Timeframe |
| `--bars` | `1000` | Candles |
| `--split` | `0.6` | Share of the history pairs are picked on; the rest confirms |
| `--min-correlation` | `0.7` | Pre-filter: return correlation on the first part |
| `--correction` | `fdr` | Allowance for how many pairs were tested: fdr \| bonferroni \| none |
| `--same-sector`, `--no-same-sector` |  | Pre-filter: only pair stocks Yahoo lists in the same sector (cuts the pairs to test) |
| `--backend` | `auto` | Where the tests run: auto (GPU if available) \| gpu \| cuda \| rocm \| cpu \| numpy |

## `tradebot halving`

Past Bitcoin halvings, how far each was from the 4-year rule, and the forecast of the next ones.

| Option | Default | Description |
|---|---|---|
| `--live`, `--no-live` |  | Also forecast from the current chain tip's pace (asks the Blockstream API) |

## `tradebot device`

Show which compute backend the optimizer will use.

## `tradebot gui`

Launch the desktop application.

## `tradebot watch`

Run the stored alerts and scheduled scans in the foreground (headless; Ctrl-C stops). Don't run it next to the app's own alert timer: both would fire the same alert.

| Option | Default | Description |
|---|---|---|
| `--poll` | `30.0` | Seconds between passes over the alerts |
| `--once` |  | One pass, then exit (for cron) |

## `tradebot bot`

Automated trading bots.

## `tradebot bot run`

Run an algorithm live against a broker (dry run unless --live or --paper). Limits are off at 0; `bot kill` stops it.

Arguments: `SYMBOL`, `ALGO`

| Option | Default | Description |
|---|---|---|
| `--broker` | `binance` | Registered broker name |
| `--tf` | `1h` | Timeframe: 1m 5m 15m 30m 1h 4h 1d 1w |
| `--stake` | `100.0` | Max quote-currency value per entry |
| `--param`, `-p` |  | Algorithm parameter, e.g. -p lower=30 |
| `--members` |  | Comma-separated members for the 'vote' ensemble, or for the 'pool' (strategies and indicators) |
| `--entry` |  | Combo entry rule (with the 'combo' strategy), e.g. "elliott_wave & close>sma(200)" or "weighted(rsi_threshold:0.5, macd_cross:0.5) >= 0.6" |
| `--exit` |  | Combo exit rule, e.g. "macd_cross \| stop_atr(2)" (default: when the entry rule stops holding) |
| `--fills` |  | Combo fills: close (stops, take-profits and limit entries checked on the close) or intrabar (filled inside the candle against its high/low, at the open on a gap; stop first when both are hit) |
| `--sizing` |  | Position sizing on top of the strategy (#147): fixed (fraction of the capital), vol (volatility target), risk (fixed risk per trade, the ATR stop distance decides the size) or kelly (fractional Kelly). Tune it with -p size.<name>=value, e.g. -p size.target_vol=0.15 |
| `--live`, `--no-live` |  | Actually place orders (default: dry run) |
| `--sandbox`, `--no-sandbox` |  | Use the exchange testnet (ccxt brokers) |
| `--once`, `--no-once` |  | Evaluate a single candle and exit |
| `--max-daily-loss` |  | Halt for the day after losing this much (quote currency) |
| `--max-drawdown` |  | Halt when equity falls this many % of the stake below its peak |
| `--max-exposure` |  | Reject entries that would push the position above this value |
| `--max-order` |  | Reject orders above this value |
| `--price-band` |  | Reject entries when the exchange price is this many % off the last close |
| `--cooldown-losses` |  | Pause entries after this many losing trades in a row |
| `--cooldown-minutes` | `60.0` | Length of that pause |
| `--close-on-halt`, `--no-close-on-halt` |  | Sell the open position when a limit halts the bot |
| `--paper`, `--no-paper` |  | Trade a simulated account that survives restarts (see `tradebot paper`) |
| `--journal`, `--no-journal` | `True` | Store every order in the trade journal (`tradebot journal`) |

## `tradebot bot kill`

Kill switch: every running `bot run` stops trading at its next check (#154).

## `tradebot bot resume`

Turn the kill switch off. A bot that already halted needs a restart.

## `tradebot bot pair-run`

Trade the gap of two assets live: long one and short the other (dry run unless --live). The short leg is a plain sell, so the broker needs margin or futures for it: give the legs as symbols it can short. With --live the open pair is saved (`bot pair-state` shows it) and picked up after a restart; a trade that didn't finish makes the next start refuse until the account has been checked.

Arguments: `LEG_A`, `LEG_B`

| Option | Default | Description |
|---|---|---|
| `--broker` | `binance` | Registered broker name |
| `--tf` | `1h` | Timeframe: 1m 5m 15m 30m 1h 4h 1d 1w |
| `--stake` | `100.0` | Quote-currency value of both legs together per entry |
| `--window` | `60` | Candles for the rolling hedge ratio and z-score |
| `--entry` | `2.0` | Open when the gap is this many σ wide |
| `--exit` | `0.5` | Close when it is back inside this many σ |
| `--stop` | `4.0` | Give up beyond this many σ |
| `--method` | `rolling` | Hedge ratio: rolling or kalman |
| `--live`, `--no-live` |  | Actually place orders (default: dry run) |
| `--sandbox`, `--no-sandbox` |  | Use the exchange testnet (ccxt brokers) |
| `--once`, `--no-once` |  | Evaluate a single candle and exit |
| `--journal`, `--no-journal` | `True` | Store every order in the trade journal (`tradebot journal`) |

## `tradebot bot pair-state`

Show what `bot pair-run --live` remembers for a pair; --keep / --forget clear an unfinished trade (#122).

Arguments: `LEG_A`, `LEG_B`

| Option | Default | Description |
|---|---|---|
| `--broker` | `binance` | Registered broker name |
| `--sandbox`, `--no-sandbox` |  | The testnet state (pair-run --sandbox) |
| `--keep`, `--no-keep` |  | Drop an unfinished trade, keep the last saved pair (it matches the account) |
| `--forget`, `--no-forget` |  | Mark the pair flat (you closed both legs at the broker) |
| `--yes`, `-y` |  | Don't ask |

## `tradebot cache`

The disk cache of candles and symbol lists.

## `tradebot cache info`

Where the cache is, how big it is and what it holds.

## `tradebot cache clear`

Delete cached data (everything unless --ohlc or --symbols); it is downloaded again when needed.

| Option | Default | Description |
|---|---|---|
| `--ohlc` |  | Only the chart data (candles) |
| `--symbols` |  | Only the symbol lists |

## `tradebot combo`

Saved strategy combos (combos.toml): entry and exit rules built from strategies and indicator conditions.

## `tradebot combo save`

Save a combo by name; it then works like any strategy: list, backtest, optimize, bot, chart, GUI. Rules: strategy names (in the exit rule: their exit), indicator conditions (close>sma(200), rsi(14)<30, elliott.state==3), & \| ! and brackets, vote(a, b, c) >= 0.6, weighted(a:0.5, b:0.3) >= 0.6, and in the exit rule the risk exits stop_pct(5) stop_atr(2) take_profit(10) take_profit_atr(3) trail_pct(5) trail_atr(3) max_hold(20); take-profits in steps scale out: take_profit(5, 0.5) \| take_profit(10). In the entry rule a limit entry: & limit_pct(1) or & limit_atr(0.5) buys below the close on the next candle. ``--fills intrabar`` fills stops, take-profits and limit entries inside the candle against its high/low. ``-p`` sets saved parameter values (e.g. tuned ones from ``optimize``).

Arguments: `NAME`

| Option | Default | Description |
|---|---|---|
| `--entry` |  | Entry rule, e.g. "elliott_wave & close>sma(200)" |
| `--exit` |  | Exit rule, e.g. "macd_cross \| stop_atr(2)" |
| `--description` |  | Shown in tradebot list and the GUI |
| `--param`, `-p` |  | Algorithm parameter, e.g. -p lower=30 |
| `--fills` | `close` | Combo fills: close (stops, take-profits and limit entries checked on the close) or intrabar (filled inside the candle against its high/low, at the open on a gap; stop first when both are hit) |

## `tradebot combo list`

The saved combos and their rules.

## `tradebot combo delete`

Delete a saved combo.

Arguments: `NAME`

## `tradebot notify`

Notification channels ([notify.*] in config.toml): Telegram, Discord, webhook, email, desktop.

## `tradebot notify test`

Send a test message to every enabled channel and report which worked.

## `tradebot bots`

Several bots at once (bots.toml next to config.toml): define, run, watch, stop (#157).

## `tradebot bots list`

The bots of bots.toml with their state, position and P&L (also those another process runs).

## `tradebot bots add`

Add a bot to bots.toml (dry run unless --mode paper/live). Parameters and rules work as in `bot run`.

Arguments: `NAME`, `SYMBOL`, `ALGO`

| Option | Default | Description |
|---|---|---|
| `--broker` | `binance` | Registered broker name |
| `--tf` | `1h` | Timeframe: 1m 5m 15m 30m 1h 4h 1d 1w |
| `--stake` | `100.0` | Max quote-currency value per entry |
| `--param`, `-p` |  | Algorithm parameter, e.g. -p lower=30 |
| `--members` |  | Comma-separated members for the 'vote' ensemble, or for the 'pool' (strategies and indicators) |
| `--entry` |  | Combo entry rule (with the 'combo' strategy), e.g. "elliott_wave & close>sma(200)" or "weighted(rsi_threshold:0.5, macd_cross:0.5) >= 0.6" |
| `--exit` |  | Combo exit rule, e.g. "macd_cross \| stop_atr(2)" (default: when the entry rule stops holding) |
| `--fills` |  | Combo fills: close (stops, take-profits and limit entries checked on the close) or intrabar (filled inside the candle against its high/low, at the open on a gap; stop first when both are hit) |
| `--sizing` |  | Position sizing on top of the strategy (#147): fixed (fraction of the capital), vol (volatility target), risk (fixed risk per trade, the ATR stop distance decides the size) or kelly (fractional Kelly). Tune it with -p size.<name>=value, e.g. -p size.target_vol=0.15 |
| `--mode` | `dry` | dry (nothing is sent), paper (the simulated account) or live (real orders, only with `bots run --live`) |
| `--sandbox`, `--no-sandbox` |  | Use the exchange testnet (ccxt brokers) |
| `--disabled`, `--no-disabled` |  | Don't start it with `bots run` (only by name) |
| `--max-daily-loss` |  | Per-bot limit, as in `bot run` (0 = off) |
| `--max-drawdown` |  | Per-bot limit, as in `bot run` (0 = off) |
| `--max-exposure` |  | Per-bot limit, as in `bot run` (0 = off) |
| `--max-order` |  | Per-bot limit, as in `bot run` (0 = off) |
| `--replace`, `--no-replace` |  | Overwrite a bot of that name |

## `tradebot bots remove`

Delete a bot from bots.toml (its saved state stays in the config folder's bots/ directory).

Arguments: `NAME`

## `tradebot bots enable`

Start this bot with `bots run` again.

Arguments: `NAME`

## `tradebot bots disable`

Leave this bot out of `bots run` (it still starts by name).

Arguments: `NAME`

## `tradebot bots limits`

Show or set the account limits all bots share. Without options it only shows them.

| Option | Default | Description |
|---|---|---|
| `--max-total-exposure` |  | Reject entries that push all bots' open positions together above this value (0 = off) |
| `--max-open-positions` |  | Reject entries while this many bots hold a position (0 = off) |
| `--max-daily-loss` |  | Halt every bot for the rest of the UTC day when all of them together lost this much (0 = off) |
| `--close-on-halt`, `--no-close-on-halt` |  | Sell the open positions when that daily limit halts the bots |

## `tradebot bots run`

Run the bot set until Ctrl+C: every bot right away, then just after each of its candles closes. Live bots are skipped unless --live. `bots stop NAME`, the app's Bots page or `bot kill` stop them.

Arguments: `NAMES` (optional)

| Option | Default | Description |
|---|---|---|
| `--live`, `--no-live` |  | Let the bots with mode = live place real orders |
| `--once`, `--no-once` |  | Evaluate one candle per bot and exit |
| `--journal`, `--no-journal` | `True` | Store every order in the trade journal |

## `tradebot bots stop`

Ask the process that runs a bot (`bots run` or the app) to stop it, within about a second.

Arguments: `NAMES` (optional)

| Option | Default | Description |
|---|---|---|
| `--all` |  | Stop every bot |

## `tradebot paper`

The simulated paper trading account.

## `tradebot paper show`

Cash, positions, P&L and the latest orders of the paper account.

| Option | Default | Description |
|---|---|---|
| `--orders` | `10` | How many recent orders to list |

## `tradebot paper reset`

Throw away positions and order history and start again with a fresh balance.

| Option | Default | Description |
|---|---|---|
| `--balance` |  | New starting balance (default: keep the current one) |
| `--yes`, `-y` |  | Don't ask |

## `tradebot journal`

The trade journal: every order the bots placed (#159).

## `tradebot journal show`

Realised P&L, fees and the latest fills with their ids, notes and tags.

| Option | Default | Description |
|---|---|---|
| `--mode` | `real` | live, paper, dry (dry runs) or real (live + paper) |
| `--symbol` |  | Only this symbol |
| `--tag` |  | Only fills with this tag |
| `--limit` | `20` | How many recent fills to list (0: none) |

## `tradebot journal note`

Set the note and/or tags of a fill.

Arguments: `FILL_ID`

| Option | Default | Description |
|---|---|---|
| `--note` |  | Note text |
| `--tags` |  | Comma-separated tags (replaces the old ones) |

## `tradebot journal export`

Export the fills (with notes, tags, realised P&L) or, with --lots, the FIFO tax lots per year as CSV.

Arguments: `PATH`

| Option | Default | Description |
|---|---|---|
| `--lots`, `--no-lots` |  | Write the FIFO tax lots instead of the fills |
| `--year` |  | With --lots: only lots closed in this year |
| `--mode` | `real` | live, paper, dry (dry runs) or real (live + paper) |

## `tradebot alert`

Price and indicator alerts and scheduled scans, delivered through the notification channels while `tradebot watch` (or the app) runs.

## `tradebot alert price`

Tell me when SYMBOL reaches a price.

Arguments: `SYMBOL`, `CONDITION`

| Option | Default | Description |
|---|---|---|
| `--tf` | `1m` | Candle the price is read from |
| `--every` | `1` | Minutes between checks, or 'close' = once per finished candle |
| `--repeat` |  | Re-arm after the condition was false again (default: fire once, then switch off) |

## `tradebot alert indicator`

Tell me when an indicator's values or alerts on SYMBOL match (all conditions).

Arguments: `SYMBOL`

| Option | Default | Description |
|---|---|---|
| `--indicator`, `-i` |  |  |
| `--param`, `-p` |  | e.g. -p period=14 |
| `--where` |  | e.g. --where "rsi < 30" |
| `--alert` |  | Indicator alert that must be on; !NAME: off |
| `--tf` | `1d` |  |
| `--every` | `close` | Minutes between checks, or 'close' = once per finished candle |
| `--repeat` |  | Re-arm after the condition was false again (default: fire once, then switch off) |

## `tradebot alert scan`

Run a scan on a schedule and report only the matches that are new since the last run.

Arguments: `SYMBOLS` (optional)

| Option | Default | Description |
|---|---|---|
| `--indicator`, `-i` | `ma_cross` |  |
| `--param`, `-p` |  |  |
| `--where` |  |  |
| `--alert` |  |  |
| `--tf` | `1d` |  |
| `--watchlist`, `-w` |  | Without symbols: only this watchlist's (default: all watchlists) |
| `--index` |  |  |
| `--every` | `close` | Minutes between checks, or 'close' = once per finished candle |

## `tradebot alert list`

The stored alerts and their state.

## `tradebot alert rm`

Delete an alert.

Arguments: `ALERT_ID`
