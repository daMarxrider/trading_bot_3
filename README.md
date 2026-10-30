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

**Picking the strategies.** Without an algorithm, `optimize` also decides *which*
strategies and indicators to use. It starts from a pool: every strategy and
indicator (plugins included), or the ones given with `--pool`. Every member gets
an on/off switch next to its parameters, and the search tunes both.
- Active strategies vote, as in `vote`, with an optimized threshold.
- An indicator can't trade on its own, so it joins as a filter with a default
  rule: a buy also needs every active filter to agree. `sma`/`ema`: close above
  the average. `rsi`/`stoch`: oversold. `macd`: MACD above zero. `bbands`/`donchian`:
  breakout. `elliott`: wave 3. Indicators without a rule (`atr`, `ichimoku`, `ssw`)
  are left out, with a note.
- `--min-members`/`--max-members` limit how many members are used; at least one
  strategy always is. Each member costs a small fitness penalty (0.01), so of two
  equally good combinations the smaller one wins.

The output lists each member as used or dropped, then the parameters of the used
ones and the command to reuse the result (`tradebot backtest … pool --members …`).
With `--method both`, the members table also shows whether GA and PSO picked the
same members. Agreement there is a good sign that the selection is real and not noise.

## Desktop app

`tradebot gui` opens these pages:

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
* **Screener**: filter any universe by technicals, volume and market cap using
  expressions or presets. Double-click a row to open it in the chart.
* **Backtest lab**: parameter forms generated from each algorithm, equity vs.
  buy & hold and monthly investing, metrics with both baselines, and the GA
  optimizer with live progress. With the **pool** algorithm, the optimizer also
  picks the members: All (default) or your own choice of strategies and
  indicator filters, with min/max members. A selection card then shows which
  members were used or dropped, and whether GA and PSO agree.
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
               Elliott waves)
  algorithms/  BaseAlgorithm + built-ins (rsi_threshold, macd_cross, ichimoku_cloud, sma_cross,
               bollinger_reversion, range, vote ensemble, elliott_wave)
  analysis/    loop-based pattern detection (Elliott waves)
  data/        providers (ccxt exchanges, yfinance, Alpha Vantage, FMP) + MarketDataEngine (fallback/stitching)
  brokers/     BaseBroker; ccxt live brokers (Binance, KuCoin, Pionex); semi-automated Scalable / Trade Republic
  backtest/    vectorized engine + metrics (Sharpe, Sortino, max drawdown, win rate, profit factor, CAGR)
  optimize/    backend selection (CUDA → ROCm → CPU), genetic algorithm, particle swarm, GA-vs-PSO comparison
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

## Extending

Built-ins and plugins register themselves when their class is defined. Put
plugin files in `~/.config/tradebot/plugins/` (or `$TRADEBOT_PLUGIN_DIR`), or
publish them under the `tradebot.plugins` entry-point group. See
`examples/plugins/donchian_breakout.py`.

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
