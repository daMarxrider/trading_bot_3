# tradebot

An extensible algorithmic trading platform for crypto and stocks: unified broker
adapters, multi-source OHLC data, pluggable indicators and strategies, a
vectorized backtester, a GPU-accelerated genetic optimizer, a PyQt6 desktop
app, and a terminal CLI.

Licensed under the **GNU GPL v3.0 or later** (see `LICENSE`).

## Disclaimer

**Not affiliated.** tradebot is independent software. It is not affiliated with, endorsed, sponsored or
approved by any broker, exchange or data provider it supports, such as Binance, KuCoin, Pionex,
Scalable Capital, Trade Republic or Yahoo. Their names and trademarks belong to their owners. tradebot
uses them only to say which services it works with.

**No warranty, no liability.** tradebot is provided "as is", without any warranty (GNU GPL v3, sections 15
and 16). It is not financial advice, and backtest or optimizer results don't predict future returns. You
use it at your own risk. To the extent permitted by law, the author is not liable for any loss. That
includes losses from bugs, wrong signals, failed, delayed, duplicated or wrongly sized orders, broker
or API outages, wrong market data, or misconfiguration. You are responsible for your accounts and API keys
and for checking every order. Run a strategy as a dry run, and on a testnet (`--sandbox`), before you use
`--live`.

The first `tradebot bot run ... --live` shows this notice and asks you to accept it. Your answer is saved
as `live_risk_accepted = "<date>"` in `config.toml`, so it asks only once. `tradebot --version` and the
desktop app's **About** button show the disclaimer too.

Market data comes from third-party services under their own terms; Yahoo Finance data, for example, is
meant for personal use. Check each broker's API terms before automating trades with it.

## Install

```sh
uv sync --extra gui --extra gpu      # or: pip install -e '.[gui,gpu]'
uv run tradebot --help
uv run tradebot gui
```

Extras: `gui` (PyQt6, pyqtgraph), `gpu` (PyTorch), `traderepublic` (pytr).
The core (CLI, backtests, CPU optimizer) needs none of them.

**Third-party tools are not included.** tradebot works with some external tools but never ships them,
neither in the package nor in the binaries. To use one, install it yourself; its own licence and terms
apply:

| Tool | Needed for | Get it |
|---|---|---|
| `pytr` | Trade Republic login | `uv sync --extra traderepublic` (source install only; the binaries don't contain it) |
| `sc` | Scalable Capital (official CLI) | [Scalable CLI releases](https://github.com/ScalableCapital/scalable-cli/releases); set its path in ⚙ Settings if it isn't on `PATH` |

## Download / run the binary

Every release on the [Releases page](https://github.com/daMarxrider/trading_bot_3/releases)
has standalone builds for Linux, Windows and macOS that need no Python. One binary is both the CLI and the
desktop app (`tradebot gui`). The builds are CPU-only (no `gpu` extra) and
**unsigned**. Each archive has a `.sha256` checksum next to it, and contains `LICENSE` plus
`THIRD_PARTY_NOTICES.txt`, the licences of all bundled libraries.

Versions follow [semantic versioning](https://semver.org): every merge into `main` is tagged
automatically, and the merged PR's label picks the bump. `release: major` marks a breaking change
(v1.4.2 → v2.0.0), `release: minor` a new feature (→ v1.5.0), and no label a bugfix or purely
cosmetic change (→ v1.4.3).

**Linux (x86_64).** Built on Ubuntu 22.04, so it runs on that and newer
distributions (glibc 2.35+).

```sh
sha256sum -c tradebot-v1.0.0-linux-x86_64.tar.gz.sha256
tar -xzf tradebot-v1.0.0-linux-x86_64.tar.gz
./tradebot/tradebot --help
./tradebot/tradebot gui
```

The GUI needs a few system libraries that desktop installs usually already
have. On Debian/Ubuntu:

```sh
sudo apt install libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 libxcb-cursor0
```

**Windows (x86_64, Windows 10/11).** Unzip
`tradebot-v1.0.0-windows-x86_64.zip`, then run `tradebot-gui.exe` for the
desktop app (no console window) or `tradebot.exe` in a terminal for the CLI:

```powershell
(Get-FileHash tradebot-v1.0.0-windows-x86_64.zip).Hash   # compare with the .sha256 file
.\tradebot\tradebot.exe --help
```

Because the binaries are unsigned, SmartScreen warns on first start ("Windows
protected your PC"). Click **More info → Run anyway**. Config, plugins and
themes live in `%USERPROFILE%\.config\tradebot\`, the same layout as on Linux.

**macOS (Apple Silicon `arm64`, Intel `x86_64`).** Unzip
`tradebot-v1.0.0-macos-arm64.zip` (or `-x86_64` on an Intel Mac) and move
`tradebot.app` to Applications. Opening the app starts the desktop app; the CLI
is inside the bundle:

```sh
shasum -a 256 -c tradebot-v1.0.0-macos-arm64.zip.sha256
/Applications/tradebot.app/Contents/MacOS/tradebot --help
sudo ln -s /Applications/tradebot.app/Contents/MacOS/tradebot /usr/local/bin/tradebot   # optional
```

The app is not signed or notarized, so Gatekeeper blocks the first start ("Apple
could not verify ..."). Try to open it once, then go to **System Settings →
Privacy & Security** and click **Open Anyway**. Alternatively, remove the
quarantine flag in a terminal with `xattr -dr com.apple.quarantine /Applications/tradebot.app`.

**Plugins in the binary.** Plugin files from `~/.config/tradebot/plugins/` (or
`$TRADEBOT_PLUGIN_DIR`) load as usual and can import any `tradebot` module and
the bundled libraries. Entry-point plugins from installed packages are not
found by the binary, because it cannot see your Python's packages; use the
plugin directory, or install tradebot from source.

**GPU backends.** The optimizer uses PyTorch, which drives NVIDIA (CUDA) and AMD
(ROCm) GPUs through the same API, so there is no vendor-specific code. For AMD,
install a ROCm build of torch, for example
`pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`.
The `gpu` extra installs the default (CUDA) build on Linux; with it an AMD card goes unnoticed and the optimizer
quietly falls back to NumPy, so reinstall torch from the ROCm index afterwards. `--backend cuda` / `--backend rocm`
(alias `amd`) require that vendor and fail with an error instead of falling back; `--backend gpu` takes either.
Algorithms marked `period_batched` (`macd_cross`, `sma_cross`) evaluate a whole population in one GPU call even when
their window lengths differ; for the other integer-only algorithms `auto` picks NumPy, which is faster for them.
Without a GPU it falls back to a multi-threaded NumPy pool. Run
`tradebot device` to see which backend is selected.

## CLI

```sh
tradebot list algorithms|indicators|brokers
tradebot chart BTC/USDT --tf 4h --bars 200 -i bbands -i rsi --algo rsi_threshold
tradebot backtest AAPL macd_cross --bars 2000 -p fast=8 -p slow=21
tradebot optimize BTC/USDT rsi_threshold --population 1024 --generations 40 --objective calmar
tradebot optimize BTC/USDT rsi_threshold --method pso          # particle swarm instead of the genetic algorithm
tradebot optimize BTC/USDT rsi_threshold --method both         # run GA and PSO, then compare them
tradebot optimize BTC/USDT                                     # also pick which strategies/indicators to use
tradebot optimize BTC/USDT --pool rsi_threshold,macd_cross,elliott_wave,sma --max-members 3
tradebot optimize AAPL,MSFT,NVDA,SAP.DE macd_cross --aggregate median   # one parameter set for several symbols
tradebot optimize -w Crypto sma_cross                          # every symbol of a watchlist (-w all: of all of them)
tradebot backtest AAPL,MSFT,NVDA macd_cross -p fast=8           # one row per symbol
tradebot optimize -w all rsi_threshold --min-symbols-beating 5 --samples 4 --sample-length 300
tradebot optimize AAPL macd_cross --walk-forward 5 --holdout 250   # re-optimize in 5 folds, then a holdout
tradebot optimize -w Crypto sma_cross --meta 20                 # also tune the optimizer's own settings
tradebot evolve AAPL,MSFT,NVDA --generations 60 --save my_rule    # evolve a whole strategy from scratch
tradebot pair KO:PEP                                           # a pair: gap z-score, hedge ratio, half-life, cointegration
tradebot pair-backtest KO:PEP --window 60 --entry 2            # long one leg, short the other, fees and borrow cost
tradebot pair-backtest KO:PEP --method kalman                  # hedge ratio from a Kalman filter instead of a rolling window
tradebot pair-scan KO PEP PG MDLZ                              # find pairs: picked on 60% of the history, confirmed on the rest
tradebot pair-scan KO PEP PG MDLZ --same-sector --correction bonferroni   # fewer pairs to test, stricter about chance finds
tradebot bot pair-run BTC/USDT:USDT ETH/USDT:USDT --once       # live pair bot (dry run; --live to trade): long one, short the other
tradebot halving --live                                        # Bitcoin halvings and the forecast of the next
tradebot bot run ETH/USDT sma_cross --broker binance --tf 1h --stake 50          # dry run
tradebot bot run ETH/USDT sma_cross --broker binance --tf 1h --stake 50 --live   # real orders
tradebot positions --broker kucoin
tradebot portfolio --broker scalable --broker binance
tradebot screen -f "rsi < 35" -f "close > sma200"
tradebot news AAPL NVDA BTC/USDT
tradebot news --watchlist Crypto   # without symbols: one watchlist (default: all of them)
tradebot cache info                 # cache folder, size, number of cached series
tradebot cache clear [--ohlc|--symbols]
```

`--bars N` returns exactly N candles, also for stocks, whose markets close on
weekends and overnight. `--start`/`--end` select a date range instead. Yahoo serves
only limited intraday history (1m: 7 days, 5m–30m: 60 days, 1h/4h: 730 days); a
request reaching further back returns what exists and prints a warning. In the
GUI, the chart and the Backtest lab have optional From/To dates that override the
candle count.

**Tab completion** (bash, fish, zsh, PowerShell). Run this once, *from the shell you want it in*:
it detects the shell and installs the completion script (bash: `~/.bash_completions/`, fish:
`~/.config/fish/completions/tradebot.fish`). Open a new shell afterwards.

```sh
tradebot --install-completion
tradebot --show-completion      # print the script instead, to install it by hand
```

TAB then completes commands and options, algorithm and indicator names (plugins
included), `-p`/`--fix` parameters of the chosen algorithm with defaults and ranges,
`-i name:param=`, `--members`, brokers, timeframes and fixed choices. Symbols
complete by ticker, case-insensitively: your watchlists first, then crypto pairs
(`sol` → `SOL/USDT`) and stocks/ETFs from Yahoo (`nvd` → `NVDA`, …). Shells match
suggestions against what you typed, so unlike the GUI you can't find a ticker by
company name here. The `tradebot` command must be on your `PATH` (an activated
venv, `uv tool install`, or the standalone binary). Lookups are cached in the OS
cache folder (e.g. `~/.cache/tradebot/`), so after the first TAB a completion
takes about half a second.

Charts are rendered with plotext, so they work in fish, bash and zsh.
`optimize` evolves parameters on the first 70 % of the data and reports the
remaining 30 % out-of-sample. Candidates with fewer than `--min-trades` round
trips are rejected.

**Baselines.** `backtest` and `optimize` (in- and out-of-sample) show the strategy
next to two things it has to beat, on the same candles and with the same fee:
- *buy & hold*: everything invested on the first candle and held.
- *monthly investing*: a fixed amount on the first candle of every month, each
  purchase paying the fee. `--dca-amount` sets the amount; by default the capital
  is split evenly over the months.

Money that goes in gradually makes total return and CAGR misleading, so every
column also shows the money-weighted return (`xirr`), and the strategy column
shows its lead over each baseline by XIRR (`vs buy & hold`, `vs monthly
investing`). When the strategy doesn't beat a baseline (out-of-sample, for
`optimize`), the output says so in red.

**GA, PSO, or both.** `--method` chooses the search:
- `ga`: genetic algorithm, the default.
- `pso`: particle swarm.
- `both`: runs GA and PSO one after the other on the same data, split, budget
  and seed, and reports how much they disagree. The report covers the
  difference per parameter (as % of its range), the in-sample score gap and
  both out-of-sample results.

The comparison ends with a verdict:
- `consistent`: both found the same region, so the optimum looks well-defined.
- `fragile`: different parameters with nearly the same score. The exact
  values are mostly noise.
- `divergent`: one method got stuck in a weaker region.

A separate warning flags overfitting when the in-sample winner does worse out
of sample. The GUI's Backtest lab offers the same options.

**Several symbols.** Parameters tuned on one chart are often curve-fitted. Give
`optimize` several symbols, comma-separated or with `--watchlist`/`-w` (repeatable;
`all` = every watchlist; the symbols argument can then be left out), and one
parameter set is scored on every symbol, each with its own 70/30 split. The
per-symbol scores become one fitness with `--aggregate`:
- `mean` (default), `median` (one outlier can't decide), `min` (the worst symbol
  counts: the most cautious), or `mean-std` (mean minus the spread between symbols).

A symbol with less than half the candles of the longest one is skipped with a
note; `--align` instead cuts every symbol to the dates all of them cover. The
output has a row per symbol (objective in-/out-of-sample, XIRR, lead over each
baseline, trades), the aggregate, on how many symbols each baseline was beaten,
and a warning when one or two symbols carry the result (the median far below the
mean). The reuse command is a `backtest` on the same symbols, which then prints
one row per symbol.

Two optional extras, off by default (CLI and Backtest lab):
- `--min-symbols-beating N`: parameter sets whose in-sample objective doesn't
  beat buy & hold's on at least N symbols are rejected, like those with too few trades.
- `--samples K --sample-length L` (random windows): each generation scores the
  candidates on K random stretches of L candles per symbol instead of all
  in-sample candles. Every window gets its own random start (different between
  windows and symbols, and new every generation), plus warm-up candles before it
  for the indicators. Because all windows have the same length, a vectorized
  strategy computes `[windows × candidates × candles]` in one batch, which suits
  the GPU. Settings that only fit one period lose. GA elites and PSO personal bests
  are scored again on every generation's windows, and the winner is chosen from
  the final population on all in-sample candles. The progress score is therefore
  noisy rather than rising steadily.

**Walk-forward validation.** One 70/30 split is a single draw: a strategy can look
good or bad depending on what the market did in that one stretch.
`--walk-forward N` runs the optimizer N times instead. Each run (a *fold*) tunes
on a training window and is scored on the next, unseen stretch; then everything
moves forward:

```
|-- train 1 --|- test 1 -|
     |-- train 2 --|- test 2 -|
          |-- train 3 --|- test 3 -|
```

- `--wf-train`/`--wf-test`: candles per training window and per test stretch.
  By default the folds fill the data and keep the `--train` ratio.
- `--anchored`: every training window starts at the first candle and grows,
  instead of rolling forward.
- `--holdout N`: the last N candles stay out of every fold. They are scored once
  at the end, with parameters tuned on the stretch right before them.

It works with `--method ga|pso|both` (with `both`, each fold keeps the in-sample
winner), several symbols (every symbol is cut at the same dates), random windows
and `--min-symbols-beating`. The output has one row per fold (dates, objective in
and out of sample, XIRR, lead over the baselines, trades), the combined
out-of-sample score (mean ± spread) and in how many folds each baseline was
beaten, then the tuned parameters per fold and how far each one moves (as % of
its range). Warnings in red flag a score that swings more between folds than its
average, a strategy that beats buy & hold in fewer than half the folds,
parameters that move across more than half their range, and a holdout that
doesn't beat buy & hold. The reuse command has the newest parameters (from the
last fold, or the ones scored on the holdout).

In the Backtest lab, *Walk-forward* in the Optimizer card is on by default with
5 rolling folds and automatic windows; *off* goes back to one 70/30 split. The
holdout is off by default (it's a fixed number of candles, so set it to fit your
data).
The plot shows every fold's test stretch shaded and numbered, with one
*walk-forward* curve through them: each stretch traded with the parameters its
fold tuned, i.e. what walk-forward trading would have earned. The holdout
continues it. The folds table below has a row per fold, the combined row and
the holdout, then the warnings and the parameters per fold. Click a fold to see
its own in- and out-of-sample curves and parameters, and *all* to return to the
overview. With several symbols, the per-symbol table shows each symbol's
walk-forward result; click one to switch the plot to it.

**Meta-search.** Which optimizer settings give the best strategy (random windows
or not, their length, `--min-symbols-beating`) is otherwise trial and error.
`--meta N` tries N combinations of them (random picks; your own settings always
first), runs a walk-forward (`--walk-forward` folds, default 5) with each and
ranks them by the mean out-of-sample score over the folds. `--meta-compute` also
tunes population and generations, `--meta-time M` stops starting new
combinations after M minutes, and Ctrl+C ends the search early (the holdout is
still scored). The objective and `--aggregate` stay your choice: they define
what "better" means, so tuning them would only pick whichever flatters the result.

Every extra tuned setting is another chance to fit noise, so the data has three
levels: the optimizer tunes on each fold's training part, the combinations are
ranked on the folds' test parts, and a holdout nothing has seen (`--holdout`, by
default as long as one fold's test stretch) is scored once with the winner.
Only that last number is the result. The output lists every combination with
its score, how far the winner is ahead of the median one, the winner's folds and
holdout, and warns when the lead is within the fold-to-fold noise, when the
winner does worse on the holdout than the median combination did on the folds,
and when it doesn't beat buy & hold on the holdout. Each combination costs a
full optimization per fold, so the command prints the number of optimizer runs
first: realistic on the GPU, slow on the CPU.

In the Backtest lab, *Meta-search* in the Optimizer card (needs walk-forward)
sets the number of combinations; *Meta: also tune* adds population and
generations. A *Stop* button next to the others ends the search early (the
winner is still scored on the holdout). The meta-search card lists every
combination, best first, with the winner's lead over the median, the holdout
result and the warnings. The plot and the folds table show the winner with its
holdout; click another combination to see its folds. *Use the winning settings*
copies them into the Optimizer card and turns the meta-search off.

**Picking the strategies.** Without an algorithm, `optimize` also decides *which*
strategies and indicators to use. It starts from a pool: every strategy and
indicator (plugins included), or the ones given with `--pool`. Every member gets
an on/off switch next to its parameters, and the search tunes both.
- Active strategies vote, as in `vote`, with an optimized threshold.
- An indicator can't trade on its own, so it joins as a filter with a default
  rule: a buy also needs every active filter to agree. `sma`/`ema`: close above
  the average. `rsi`/`stoch`: oversold. `macd`: MACD above zero. `bbands`/`donchian`:
  breakout. `elliott`: wave 3. `halving`: a window of days after a
  Bitcoin halving. Indicators without a rule (`atr`, `ichimoku`, `ssw`)
  are left out, with a note.
- `--min-members`/`--max-members` limit how many members are used; at least one
  strategy always is. Each member costs a small fitness penalty (0.01), so of two
  equally good combinations the smaller one wins.

The output lists each member as used or dropped, then the parameters of the used
ones and the command to reuse the result (`tradebot backtest … pool --members …`).
With `--method both`, the members table also shows whether GA and PSO picked the
same members. Agreement there is a good sign that the selection is real and not noise.

## Evolving strategies from scratch

`tradebot evolve` doesn't tune a strategy you already have. It writes one: the
entry and exit rules are evolved with genetic programming, including which
indicators they use, how those are combined, and every number in them. A result
looks like this:

```
entry = crosses_above(ema(close, 13), sma(close, 41)) and rsi(9) < 58
exit  = close < shift(lowest(low, 7), 1) or bars_in_trade >= 30
```

```sh
tradebot evolve AAPL                                   # one symbol, default settings
tradebot evolve AAPL,MSFT,NVDA --generations 60 --population 500
tradebot evolve -w Crypto --tf 4h --indicators rsi,ema,atr --max-depth 4
tradebot evolve AAPL --permutations 5                  # also run it 5 times on shuffled prices
tradebot evolve AAPL --save my_rule                    # keep it as a plugin algorithm
```

**Building blocks.** The rules are trees made of:
- the candles (open, high, low, close) and every indicator output, plugins
  included. An output that looks into the future is left out automatically,
  like the Elliott zigzag: the evolver checks that its values don't change
  when later candles are added.
- `sma`, `ema`, `highest`, `lowest`, `shift`, `max`, `min`, `if_else`,
  `change(x, n)` (relative change), `gap(a, b)` (a/b − 1), `volatility(n)`,
  `price + k × atr(n)`;
- comparisons (`>`, `<`, `crosses_above`, `crosses_below`), `and`, `or`, `not`;
- in exit rules, the open trade: `bars_in_trade`, `stop_loss`, `take_profit`
  and `trailing_stop` (fractions of the price; `--no-state` turns them off).

Every part has a unit: a price, a 0–100 oscillator, a relative change, a
price distance (ATR), or an indicator's own unit like MACD. Only like is
compared with like, and a price is never compared with a fixed number, because
"close > 182.5" means nothing on another symbol. Every tree the search builds is
therefore valid, and it means the same on every chart. Plugin indicators can
declare their outputs' units with `units = {"value": "osc"}`. Otherwise
overlays count as prices, and indicators whose `levels` lie within 0–100
count as oscillators.

**The search.** It starts with 300 random programs (`--population`). In each
of 40 generations (`--generations`) it keeps the best few, and breeds the rest:
- by swapping parts between two good programs, or a whole entry or exit rule;
- by mutation: a new random part, `>` ↔ `<` or `sma` ↔ `ema`, a nudged number,
  or a part replaced by one of its own sub-parts.

`--max-depth` (6) and `--max-size` (40 building blocks per rule) cap the size, and
every building block costs a little fitness (`--parsimony`, 0.002). Of two equally
good rules, the shorter one wins. Equivalent rules are recognised (`a and b` =
`b and a`, `x > x` is never true, …), so each is scored once.

**Keeping it honest.** A search this large finds patterns in pure noise, so the
data is split by date, all symbols at the same dates:

```
|---------- train (3 segments) ----------|-- validation (20 %) --|-- test (20 %) --|
```

- The fitness only sees the training part. It's scored in `--segments`
  consecutive stretches on every symbol and combined with `--aggregate`
  (mean, median, min or mean−std), so a rule has to work in every stretch and
  on every symbol. Programs with fewer than `--min-trades` round trips on a
  symbol are rejected.
- The 30 best programs ever seen are then compared on the validation part, and
  the best one there is the result. Evolution never sees validation scores.
- The result is scored once on the test part (`--test`), which no step of the
  search has seen, next to buy & hold and monthly investing.
- `--permutations N` runs the same search N more times on shuffled candles:
  the same moves in random order, so there is no pattern left to find. The
  p-value is the share of runs (the real one included) that scored at least as
  well on the test part. A large p means the evolver finds as much in noise.

The output shows the rules, a row per symbol (score on train, validation and
test, the test XIRR, the lead over each baseline, trades), the best programs
with their validation scores, and warnings in red. A warning appears when the
result doesn't beat buy & hold on the test part, scores much worse on unseen data
than in training, makes fewer than 3 test trades, depends on one or two symbols,
or fails the permutation test. Every run prints its `--seed`, so it can be
repeated.

**From scratch (`--from-scratch`, the "From scratch" box in the GUI).** No
ready-made indicators at all: rules are built from the candles and low-level
maths, so the evolver can invent RSI, Stochastic, Bollinger, ATR or MACD itself,
or something nobody has named yet. Runtime doesn't matter in this mode; it
trades speed for expressiveness and safeguards.

- Building blocks: `diff(x, n)`, `pos(x)` / `neg(x)` (up and down moves),
  `rolling_sum`, `rolling_std`, `wilder` (Wilder's smoothing), `sma`, `ema`,
  `highest`, `lowest`, `shift`, `abs`, free constants, `100 * ratio` and
  `squash(ratio)` (onto 0–100), and `+ − × ÷`.
- The arithmetic is checked by dimensional analysis, like units in physics:
  `price − price` is a distance, `price ± distance` a price, `price / price` and
  `distance / price` are ratios, `price × ratio` a price. `price + price` and
  `price × price` can't be built, so a rule still means the same on every
  symbol. A division of a non-zero value by zero rejects the program; 0/0
  (a flat stretch) is a gap, like a warm-up.
- RSI(14), for example, is
  `100 * wilder(pos(diff(close, 1)), 14) / (wilder(pos(diff(close, 1)), 14) + wilder(neg(diff(close, 1)), 14))`.
  The tests build all five indicators like this and check that they equal
  the built-in ones. A slow test hides an `RSI(14) < 30` rule in synthetic
  prices (`pytest --slow`): evolve gets close (rules that catch most of its
  trades) but doesn't find RSI itself yet.
- **Evolved building blocks**: every 5 generations, parts that keep showing up in
  the best programs become named modules (`m1`, `m2`, …), used as a single
  building block from then on. Equal numbers inside one become one parameter,
  so `m1(n)` above would be RSI with its window `n`. The rules print with a
  `where` section defining them, and "Save as plugin" writes them as functions.
- Limits are raised to depth 12 and 100 building blocks. Windows and lags are
  searched on a log scale, and comparison levels for ratios span −1 to 1, since
  ratios here include normalised oscillators like
  `wilder(diff(close, 1), 14) / wilder(abs(diff(close, 1)), 14)` (that is
  RSI(14) / 50 − 1, so `< −0.4` means RSI below 30).
- **Islands** (`--islands`, default 4): the population is split into groups that
  evolve apart and pass their 2 best on to the next group every 10 generations,
  so several ideas survive long enough to be refined. The best 3 programs of
  every island and generation get their numbers hill-climbed.
- Safeguards, on by default: 5 shuffled-price runs; 3 walk-forward folds
  (`--folds`), where the search runs again on rolling windows and each result is
  scored on the stretch after its window; the **deflated Sharpe ratio**, which
  corrects the training Sharpe ratio for the number of different strategies
  tried (warning below 0.95); and deduplication, so strategies that trade
  exactly alike count once.

```sh
tradebot evolve BTC/USDT --from-scratch --population 1000 --generations 100
tradebot evolve AAPL,MSFT --from-scratch --permutations 19 --folds 5
```

**Using the result.** `--save NAME` writes the program as a plugin file
(`~/.config/tradebot/plugins/NAME.py`, or `--plugin-dir`). It's ordinary
strategy code with the rules in its docstring, so it works in `backtest`, the
chart, the Backtest lab and the bot, and you can read and edit it. Every number
in the rules becomes a parameter (`n1`, `n2`, … for whole numbers, `c1`, … for
the rest). `tradebot optimize SYMBOL NAME` can then fine-tune them, with
walk-forward if you like. Exits that look at the open trade are evaluated trade
by trade on the CPU. Everything else is batched like any strategy.

The search itself runs on the CPU with NumPy. Different programs can't share
one batch, but parts that many programs share (such as `rsi(14)` or
`sma(close, 50)`) are computed once and cached. A run with the defaults on
2,500 daily candles of one symbol takes under a minute; time grows with the
number of symbols, and `--permutations N` repeats the whole search N times.

## Desktop app

`tradebot gui` opens these pages:

* **Scanner**: runs one indicator (built-in, community, or your own Python script written in an
  in-app editor with live reload) over a watchlist or a market index, filtered on its values and
  alerts. See [docs/scanner.md](docs/scanner.md); the CLI twin is `tradebot scan`.
* **Chart**: the symbol field suggests matches while you type: your watchlists and
  portfolio first, then stocks, ETFs and indices by ticker or company name (Yahoo
  search), and crypto pairs from the exchange (e.g. `sol` → `SOL/USDT`). Pick one
  to load it. The Backtest lab's symbol field works the same way. Candles or line,
  any indicator as an overlay or in its own pane,
  **Loadouts ▾** (top right) saves the current indicators and signal under a name
  you choose and applies a saved set in one click (✕ deletes one after asking).
  The chart also reopens with the symbol, timeframe, indicators and signal you
  had when you closed it, saved or not.
  buy/sell markers for any algorithm, a crosshair with an OHLC readout, and
  drawing tools (trendline, support/resistance level, zone rectangle). The ↕
  button in the bottom-left corner (on by default) fits the price axis and
  indicator panes to the candles in view while you scroll or zoom. While it is on,
  dragging, scrolling and pinching only move the time axis, and not past the
  first or last candle, so the chart always stays in view. Switch it off to move
  the price axis freely. The app remembers the setting between sessions.
  The **log** button next to it switches the price pane to a logarithmic scale, where equal percentage moves
  have equal height. Use it for assets that rose a lot, like Bitcoin. The axis still shows prices, and
  overlays, markers and drawings stay at their prices. Indicator panes stay linear. The app remembers this
  setting too. Data with prices at or below 0 is shown on a linear scale.
* **Chart sidebar**: a panel at the right of the chart with one page per category. The topmost button
  collapses it to a narrow strip (») and expands it again («). The category buttons below it switch
  pages, and also open a collapsed sidebar. The app remembers whether it was open and which category
  was shown.
  * **📰 News**: headlines for the symbol on the chart, for stocks and crypto pairs alike (Yahoo Finance).
    They follow the chart when you switch symbols and refresh every 5 minutes. They load only while the
    sidebar shows them. Click a headline to open it in the browser.
  * **⭐ Watchlists**: one or more named lists of symbols, each with its full name, last price and the
    day's change. Pick a list in the box at the top; ＋ creates one, ✎ renames it, 🗑 deletes it (after
    asking; the last list stays). Add symbols with the field (same suggestions as the chart's), with
    **Add current symbol**, or from the chart's right-click menu (**Add to watchlist**). Click a row to
    chart it; the charted symbol is highlighted. Remove a row with its ✕ (shown on hover) or the Delete
    key; reorder by drag and drop. Right-click a row (long press on a touchscreen) for Move up/down and
    Remove. Prices come from the cache first, then refresh every minute while the list is shown. The
    lists are saved in `config.toml` (see [Configuration](#configuration)), the shown list in the app's
    settings.
* **Pairs**: two assets normalised on one chart, their gap as a z-score, correlation, hedge ratio,
  half-life, an Engle-Granger cointegration test and a warning when a pair seems to have stopped working.
  The hedge ratio comes from a rolling regression or a Kalman filter.
  `A:B` (e.g. `KO:PEP`) is also a symbol everywhere: candles of the ratio, converted into one currency.
* **Screener**: filter any universe by technicals, volume and market cap using
  expressions or presets. Double-click a row to open it in the chart.
* **Backtest lab**: parameter forms generated from each algorithm, equity vs.
  buy & hold and monthly investing, metrics with both baselines, and the GA
  optimizer with live progress. With the **pool** algorithm, the optimizer also
  picks the members: All (default) or your own choice of strategies and
  indicator filters, with min/max members. A selection card then shows which
  members were used or dropped, and whether GA and PSO agree. The symbol field
  takes several symbols (comma-separated, completion works on the last one), and
  its ☰ button fills in a watchlist or all of them. With several symbols, Run
  backtest and Optimize add a per-symbol table (with the aggregate and the
  baselines beaten); click a row to see that symbol's curves and metrics.
* **Evolve**: `tradebot evolve` with a form (see
  [Evolving strategies from scratch](#evolving-strategies-from-scratch)): the
  symbols (☰ for a watchlist), the search settings, the train/validation/test
  split and the shuffled-price runs, and which indicators the rules may use.
  Stop ends a run early. The result shows:
  - the rules, with the dates of the parts and the seed;
  - the warnings in red;
  - the equity of the train, validation and test parts (the latter two shaded)
    against buy & hold and monthly investing;
  - the best score per generation in training and on validation (training rising
    while validation stays flat means overfitting);
  - a row per symbol (click one to plot it);
  - the best strategies in training, with an arrow at the one picked on validation.

  Both plots have the chart's ↕ auto-fit button in their bottom-left corner. It is
  on by default and fits the value axis to the curves in view while you zoom or
  pan. Double-click or double-tap a plot to see all of it again.

  *Save as plugin…* keeps the result under a name you choose. It appears in the
  Backtest lab's and the chart's strategy lists right away. *Show on chart*
  shows its signals without saving it.
* **Resizing (Backtest lab and Evolve)**: drag the dividers between the result
  panes, and between the settings column and the results. Tall cards scroll
  inside their pane instead of squeezing the plot. The sizes are remembered,
  and double-clicking a divider resets them.
* **Portfolio & dividends**: total equity, allocation donuts, and per-holding
  and total current dividend yield, trailing annual yield and yield on cost.
* **News**: headlines for the watchlists' and portfolio symbols, auto-refreshed
  every 5 minutes.

**Touchscreens.** The charts support the common touch gestures:

| Gesture | Action |
|---|---|
| Pinch / spread | Zoom around your fingers. A horizontal pinch zooms the time axis, a vertical pinch zooms the price axis. |
| One- or two-finger drag | Pan |
| Flick | Pan with momentum |
| Tap | Crosshair and OHLC readout; also places drawing points |
| Double-tap | Reset the view |
| Long press | Context menu (also on right-click): reset zoom, add a level at the touched price, drawing tools, candles/line |

Lists and tables scroll with a kinetic flick. With a touchscreen present, controls automatically get finger-sized hit targets. Override this with `touch_mode = "on"` or `"off"` in the config.

**⚙ Settings** (bottom left of the sidebar) opens the settings, with the categories on the left:
* **Appearance**: the theme. It applies right away.
* **Integrations**:
  * *Third-party tools*: the path to each external tool tradebot can use, such as the Scalable Capital
    CLI (`sc`). Leave it empty to find the tool on `PATH`. Each tool shows whether it was found, and a
    download link if it wasn't.
  * *Broker API keys*: the credentials of every broker (API key, secret, passphrase, holdings CSV, …),
    saved under `[brokers.<name>]`. They're stored in plain text. A key set as an environment variable
    (`TRADEBOT_<BROKER>_<KEY>`) takes precedence and is shown greyed out. Use API keys without
    withdrawal rights.
* **Data & cache**: switch the disk cache on or off and set its maximum size. The page shows how much
  chart data and how many symbol lists are cached, and where. *Clear chart data*, *Clear symbol lists*
  and *Clear all* delete them after asking.

Changes are saved to `config.toml` right away.

Themes: **Glass** (translucent frosted panels over a soft gradient backdrop),
**Modern Dark** and **Light**. You can add your own by placing JSON files in
`~/.config/tradebot/themes/`:

```json
{"name": "Midnight", "base": "dark", "accent": "#ff6ad5", "bg": "#0b0b1a"}
```

## Configuration

`~/.config/tradebot/config.toml` (override the path with `$TRADEBOT_CONFIG`):

```toml
theme = "glass"

[watchlists]                # edited in the chart sidebar, or here by hand
Main = ["AAPL", "MSFT", "SAP.DE", "BTC/USDT", "ETH/USDT"]
"Long term" = ["SAP.DE", "ETH/USDT"]   # quote names with spaces

[brokers.binance]
api_key = "..."
secret = "..."

[brokers.kucoin]
api_key = "..."
secret = "..."
password = "..."            # KuCoin API passphrase

[brokers.scalable]
holdings_csv = "~/finance/scalable.csv"   # columns: symbol,quantity,avg_cost

[tools]
sc = "/opt/scalable/sc"     # optional; default: found on PATH
```

The settings dialog writes `theme`, the `[tools]` paths and the broker keys to this file, so the app
starts with them next time. Only those lines change; your comments and other
settings stay as they are, and a missing file is created. An empty field
removes its line.

Watchlists work the same way. An older config's top-level `watchlist = [...]` still works: it shows up
as the list "Main" and moves to `[watchlists]` the first time you change a watchlist in the app.
Suggestions, the screener's default symbols, the News page, the CLI's `screen`/`news` defaults and the
cache warm-up at startup use the symbols of all lists.

**Cache.** Candles and symbol lists are kept on disk, so the app and the CLI start without waiting for the
network:

* **Candles**: one file per symbol and timeframe. The chart shows the cached candles right away, then fetches
  only the newest ones in the background. The last two cached candles are always fetched again, because the
  newest may not have been complete yet.
* **Adjusted history**: if a fresh candle differs from the cached one by more than 0.5 %, the history was
  split- or dividend-adjusted, and the series is downloaded again in full.
* **Offline**: the cached candles are shown, with a note that they may be outdated.
* **Symbol search**: crypto pairs are kept for 24 h; an older list still answers right away while a fresh
  one loads. Yahoo results are kept per query for 7 days (at most 5,000 queries). Instrument names are kept
  for 30 days. The CLI's tab completion and the app share these files.
* **Cleanup**: once a day at most, in the background. Series not used for 60 days are deleted, and 1m–30m
  candles after 14 days. Over the size cap, the least recently used series go first, until the cache is at
  80 % of the cap. A series keeps at most 50,000 candles.

The cache lives in the OS cache folder (`~/.cache/tradebot/`, `~/Library/Caches/tradebot/`,
`%LOCALAPPDATA%\tradebot\Cache\`, or `$TRADEBOT_CACHE_DIR`), and the whole folder is safe to delete.
In `config.toml`:

```toml
[cache]
enabled = true    # false: keep candles in memory only
max_mb = 500
```

Settings the app remembers by itself (such as the chart's auto-fit toggle) are
saved in `settings.json` in the OS config folder, not in `config.toml`:
`$XDG_CONFIG_HOME/tradebot/` (usually `~/.config/tradebot/`) on Linux,
`~/Library/Application Support/tradebot/` on macOS and `%APPDATA%\tradebot\` on
Windows. Delete the file to reset them.

Secrets can also come from the environment as `TRADEBOT_<BROKER>_<KEY>`, for
example `TRADEBOT_BINANCE_SECRET`. The optional data providers read
`ALPHAVANTAGE_API_KEY` and `FMP_API_KEY`.

## Architecture

```
src/tradebot/
  core/        ops (NumPy/torch time-series primitives), OHLC container, registry + plugin loader, models
  indicators/  BaseIndicator + built-ins (SMA, EMA, RSI, MACD, Bollinger, ATR, Stochastic, Donchian, Ichimoku,
               Elliott waves, Bitcoin halving cycle)
  algorithms/  BaseAlgorithm + built-ins (rsi_threshold, macd_cross, ichimoku_cloud, sma_cross,
               bollinger_reversion, range, vote ensemble, elliott_wave, halving_cycle)
  analysis/    loop-based pattern detection (Elliott waves), Bitcoin halving schedule and forecast
  data/        providers (ccxt exchanges, yfinance, Alpha Vantage, FMP) + MarketDataEngine (fallback/stitching)
  brokers/     BaseBroker; ccxt live brokers (Binance, KuCoin, Pionex); semi-automated Scalable / Trade Republic
  backtest/    vectorized engine + metrics (Sharpe, Sortino, max drawdown, win rate, profit factor, CAGR)
  optimize/    backend selection (CUDA → ROCm → CPU), genetic algorithm, particle swarm, GA-vs-PSO comparison
  evolve/      genetic programming of whole strategies: typed primitives, trees, search, plugin export
  live/        TradingBot (same execution model as the backtester)
  portfolio.py, screener.py, news.py, config.py
  cli/         typer + rich + plotext
  gui/         PyQt6 + pyqtgraph
```

**One execution model.** An algorithm returns a target position per bar (1
long, 0 flat, −1 short when allowed). The position decided at bar *t*'s close
earns bar *t+1*'s return, and fees are charged on every position change. The
backtester, the optimizer and the live bot all use this model, so there is no
look-ahead and live behaviour matches backtests.

**OHLC only.** Indicators, charts and backtests use open, high, low and close.
Volume is kept only for the screener.

**Data fallback.** Providers are tried in order. When a provider covers only
part of the requested range, only the missing head or tail is requested from
the next one. Because prices are assumed identical across venues, the pieces
can be stitched together; for example, Yahoo history fills in before a coin was
listed on Binance.

**Brokers.** Every broker has the same `get_ticker`, `get_klines`,
`place_order`, `cancel_order` and `get_balance` methods, plus an execution
mode:

* `live`: orders go to the venue's API. All crypto venues use ccxt.
* `semi_auto`: there is no trading API (Scalable Capital, Trade Republic).
  Orders become `OrderIntent`s for you to execute by hand, and holdings are
  read from a CSV.
* `read_only`: tracking only.

## Elliott waves

**Indicator `elliott`.** It finds swings with an ATR-based zigzag (`swing_atr` sets the minimum swing size). It then checks every swing sequence against the hard Elliott rules:
- wave 2 never retraces more than 100% of wave 1;
- wave 3 is never the shortest of waves 1, 3 and 5;
- wave 4 doesn't overlap wave 1;
- in a correction, C extends beyond A.

Each match gets a Fibonacci conformity score from 0 to 1. Charts draw the zigzag with labels 0–5 and A–C. These labels are a **hindsight** view, so the indicator also provides a **causal** `state` output for trading and screening:
- `3`, `4`, `5`: that wave is in progress;
- `10`: the impulse is complete;
- `11`–`13`: waves A–C of the correction;
- negative values: the same for a bearish structure.

`state` only uses swings that were already confirmed at each bar, for example `tradebot screen -f "elliott_state == 3"` after adding it as an extra screener column.

**Strategy `elliott_wave`** (long-only):
- **Wave-3 setup:** after a valid wave 1–2 whose wave-2 retracement lies within `wave2_min`–`wave2_max`, it buys the breakout above the wave-1 high.
  - stop: below the wave-2 low, minus `stop_atr` × ATR
  - target: `target_ext` × wave 1
  - also exits when the wave-3 top gets confirmed
- **Wave-5 setup** (`trade_wave5`): the same logic after a valid wave 1–4.
- **Optimization:** all parameters can be tuned with `tradebot optimize … elliott_wave --method both`.
- **Speed:** the strategy needs a loop over the bars, so the optimizer evaluates it on the CPU (about 3.5 ms per candidate for 1,500 candles).

## Bitcoin halving cycle

Every 210,000 blocks the Bitcoin block reward halves. At the target of 600 s per block that is 1458.33 days,
the "4-year rule" (`nSubsidyHalvingInterval` and `nPowTargetSpacing` in Bitcoin's `chainparams.cpp`). Blocks
come faster while the hashrate grows, so every halving so far came early, by 1.3 % to 9.6 %.
`tradebot halving` shows the past halvings and how each estimator would have predicted them from the earlier
ones. The best (the average time per block of all completed epochs) forecasts the next halving; `--live` adds a
forecast from the current chain tip's pace.

**Indicator `halving`** is a calendar clock, so it works on any timestamped candles (also as a crypto-market
cycle clock on other coins):
- `days`: days since the latest halving;
- `days_to_next`: days until the next halving *as predicted at that bar from the earlier halvings only*, so a
  backtest never sees the future (negative while a halving is later than predicted);
- `phase` (not drawn): `days` as a fraction of the predicted cycle, 0 at a halving and about 1 at the next.

**Strategy `halving_cycle`** is long from `buy_day` to `sell_day` days after a halving. With `sell_day` below
`buy_day` the trade spans the halving. The defaults (buy at day 850, sell at day 530) come from the past
cycles: tops 371–548 days after a halving, bottoms 777–924. With only four cycles so far, optimized days fit
the past easily: use `--walk-forward` and compare with buy & hold.

## Extending

Built-ins and plugins register themselves when their class is defined. Put
plugin files in `~/.config/tradebot/plugins/` (or `$TRADEBOT_PLUGIN_DIR`), or
publish them under the `tradebot.plugins` entry-point group. See
`examples/plugins/donchian_breakout.py`; indicator scripts can be written and reloaded live in the
Scanner's editor (`examples/plugins/near_high.py`, [docs/scanner.md](docs/scanner.md)).

**Algorithm**

```python
from tradebot.algorithms.base import BaseAlgorithm
from tradebot.core import ops
from tradebot.core.registry import IntParam, Param
from tradebot.indicators.builtin import RSI

class RsiDip(BaseAlgorithm):
    name = "rsi_dip"
    params = {"period": IntParam(14, 2, 50), "level": Param(25, 5, 45)}

    def signals(self, o, p):
        rsi = RSI.compute(o, period=p["period"])["value"]
        return ops.latch(rsi < p["level"], rsi > 50)
```

If you only use `tradebot.core.ops` and arithmetic, the same code runs in
backtests, the live bot and the batched GPU optimizer. The optimizer passes
float parameters as `[population, 1]` arrays, and integer parameters are always
plain `int`.

**Indicator.** Subclass `BaseIndicator`: set `name`, `params` and `outputs`,
set `overlay=False` for its own pane, and add `levels` for reference lines.
Then implement `compute(cls, o, **params) -> dict`.

To choose how each output is drawn, set `styles`:

```python
from tradebot.indicators import Style

styles = {
    "fast": Style(color="#ffd166", width=2),      # any CSS colour, rgba(r, g, b, a) too
    "slow": Style(color="up", dash="dash"),       # a theme colour: follows the theme
}
```

Theme colours are `up`, `down`, `accent`, `muted`, `text` and `series[N]` (the
palette's N-th colour). `dash` is `"dash"` or `"dot"`. A `*_hist` output draws as
bars in its colour. Outputs without a colour take the theme palette, and keep
their colour when other indicators are added or removed. The chart's value
readout uses the same colours.

**Broker.** For any ccxt exchange this is two lines:

```python
class Bitstamp(CcxtBroker):
    name = "bitstamp"
    exchange_id = "bitstamp"
```

For venues without an API, subclass `ManualExecutionBroker` or
`HoldingsFileBroker`.

## Development

```sh
uv run pytest
uv run pytest --slow                   # also the minutes-long tests (CI runs them on PRs labelled `slow tests`)
uv run python scripts/bench_chart.py   # chart redraw and pan times for 500 / 5,000 / 50,000 candles
```

Don't add third-party tools or broker assets to the repo or the binaries. Integrations call a tool the
user installed (found on `PATH` or via a configured path) and fail with a clear message and a download
link if it's missing. Add a row for the tool to the table under *Install*. Broker names appear only as
plain text, never as logos.

The tests cover indicator accuracy against pandas reference implementations,
backtest math on hand-checked series, NumPy/torch-CPU/CUDA parity of batched
fitness, the broker abstractions, data-engine fallback, plugin loading, the
live bot, portfolio and dividend metrics, the screener, news parsing, and
offscreen rendering of every GUI theme.
