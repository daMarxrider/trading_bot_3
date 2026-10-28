# tradebot

An extensible algorithmic trading platform for crypto and stocks: unified broker
adapters, multi-source OHLC data, pluggable indicators and strategies, a
vectorized backtester, a GPU-accelerated genetic optimizer, a PyQt6 desktop
app, and a terminal CLI.

Licensed under the **GNU GPL v3.0 or later** (see `LICENSE`).

## Install

```sh
uv sync --extra gui --extra gpu      # or: pip install -e '.[gui,gpu]'
uv run tradebot --help
uv run tradebot gui
```

Extras: `gui` (PyQt6, pyqtgraph), `gpu` (PyTorch), `traderepublic` (pytr).
The core (CLI, backtests, CPU optimizer) needs none of them.

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
tradebot bot run ETH/USDT sma_cross --broker binance --tf 1h --stake 50          # dry run
tradebot bot run ETH/USDT sma_cross --broker binance --tf 1h --stake 50 --live   # real orders
tradebot positions --broker kucoin
tradebot portfolio --broker scalable --broker binance
tradebot screen -f "rsi < 35" -f "close > sma200"
tradebot news AAPL NVDA BTC/USDT
```

Charts are rendered with plotext, so they work in fish, bash and zsh.
`optimize` evolves parameters on the first 70 % of the data and reports the
remaining 30 % out-of-sample. Candidates with fewer than `--min-trades` round
trips are rejected.

## Desktop app

`tradebot gui` opens these pages:

* **Chart**: candles or line, any indicator as an overlay or in its own pane,
  buy/sell markers for any algorithm, a crosshair with an OHLC readout, and
  drawing tools (trendline, support/resistance level, zone rectangle).
* **Screener**: filter any universe by technicals, volume and market cap using
  expressions or presets. Double-click a row to open it in the chart.
* **Backtest lab**: parameter forms generated from each algorithm, equity vs.
  buy & hold, metrics, and the GA optimizer with live progress.
* **Portfolio & dividends**: total equity, allocation donuts, and per-holding
  and total current dividend yield, trailing annual yield and yield on cost.
* **News**: headlines for watchlist and portfolio symbols, auto-refreshed
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

Themes: **Glass** (translucent frosted panels over a soft gradient backdrop),
**Modern Dark** and **Light**. You can add your own by placing JSON files in
`~/.config/tradebot/themes/`:

```json
{"name": "Midnight", "base": "dark", "accent": "#ff6ad5", "bg": "#0b0b1a"}
```

## Configuration

`~/.config/tradebot/config.toml` (override the path with `$TRADEBOT_CONFIG`):

```toml
watchlist = ["AAPL", "MSFT", "SAP.DE", "BTC/USDT", "ETH/USDT"]
theme = "glass"

[brokers.binance]
api_key = "..."
secret = "..."

[brokers.kucoin]
api_key = "..."
secret = "..."
password = "..."            # KuCoin API passphrase

[brokers.scalable]
holdings_csv = "~/finance/scalable.csv"   # columns: symbol,quantity,avg_cost
```

Secrets can also come from the environment as `TRADEBOT_<BROKER>_<KEY>`, for
example `TRADEBOT_BINANCE_SECRET`. The optional data providers read
`ALPHAVANTAGE_API_KEY` and `FMP_API_KEY`.

## Architecture

```
src/tradebot/
  core/        ops (NumPy/torch time-series primitives), OHLC container, registry + plugin loader, models
  indicators/  BaseIndicator + built-ins (SMA, EMA, RSI, MACD, Bollinger, ATR, Stochastic, Donchian, Ichimoku)
  algorithms/  BaseAlgorithm + built-ins (rsi_threshold, macd_cross, ichimoku_cloud, sma_cross,
               bollinger_reversion, range, vote ensemble)
  data/        providers (ccxt exchanges, yfinance, Alpha Vantage, FMP) + MarketDataEngine (fallback/stitching)
  brokers/     BaseBroker; ccxt live brokers (Binance, KuCoin, Pionex); semi-automated Scalable / Trade Republic
  backtest/    vectorized engine + metrics (Sharpe, Sortino, max drawdown, win rate, profit factor, CAGR)
  optimize/    backend selection (CUDA → ROCm → CPU) + genetic algorithm
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
```

The tests cover indicator accuracy against pandas reference implementations,
backtest math on hand-checked series, NumPy/torch-CPU/CUDA parity of batched
fitness, the broker abstractions, data-engine fallback, plugin loading, the
live bot, portfolio and dividend metrics, the screener, news parsing, and
offscreen rendering of every GUI theme.
