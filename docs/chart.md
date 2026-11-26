# Chart

The **📈 Chart** page is the first page of the desktop app (`tradebot gui`). The terminal twin is
`tradebot chart SYMBOL` (plotext, works in bash, zsh and fish).

## Loading a symbol

Click the symbol field, or just start typing with the chart under the cursor, and the **symbol dialog** opens. It
searches your watchlists and portfolio first, then stocks, ETFs, indices, forex, futures and Treasury yields by ticker
or name (Yahoo search), and the exchange's crypto pairs (`sol` finds `SOL/USDT`, `btcusd` finds `BTC/USD`). The rail
on the left filters by category (All, Stocks, Forex, Crypto, Indices, Futures, Bonds; ETFs and funds count as
stocks) or shows one of your lists (recent symbols, each watchlist). Each row has the last price and the day change.
↑/↓ move, Enter loads, Tab switches category, Esc closes. To chart a ratio, press `/` on a symbol (or its ÷ button)
and pick the divisor: `BTCUSD/ETHUSD` typed the way Scalable writes it works too. For now both sides must be crypto
pairs, or neither. Fields that take several symbols (Backtest, Evolve) open the dialog with their search icon; it
stays open until Esc, so you can add several. With the keyboard in a symbol field (Tab), typing still suggests
matches inline. The **timeframe** and **candles** boxes reload the chart; optional **From/To**
dates override the candle count. `A:B` (for example `KO:PEP`) is a symbol too: candles of the ratio, converted into
one currency (see [Pairs](pairs.md)).

The chart reopens with the symbol, timeframe, indicators and signal you had when you closed it, saved or not.

**Live** (next to Load) keeps the last candle moving with the market: its high, low and close follow every price, and
a new candle appears when the next one opens; indicators and signals redraw with it, at most once a second. Crypto
streams from the exchange's websocket, stocks and ETFs from Yahoo's price stream during market hours. When the
websocket isn't available or keeps dropping, the chart polls for prices every few seconds instead and the button's
icon turns into a warning sign (its tooltip says why); it tries the websocket again after five minutes. Live follows
one symbol, not a ratio (`A:B`), and switches to the new symbol or timeframe when you load another. Details:
[Streaming prices](streaming.md).

## Candles, line, volume

* **Line** toggles between candles and a line chart. Right-click offers the same.
* **vol** (bottom-left of the plot) shows a volume pane under the candles when the data has volume; the right-click
  menu can also draw a volume overlay on the price pane.
* **↕** (bottom-left, on by default) fits the price axis and the indicator panes to the candles in view while you
  scroll or zoom. While it is on, dragging, scrolling and pinching only move the time axis, and not past the first or
  last candle, so the chart always stays in view. Switch it off to move the price axis freely.
* **log** switches the price pane to a logarithmic scale, where equal percentage moves have equal height. Use it for
  assets that rose a lot, like Bitcoin. The axis still shows prices, and overlays, markers and drawings stay at their
  prices. Indicator panes stay linear. Data with prices at or below 0 is shown on a linear scale.

The app remembers all three toggles between sessions.

## Chart types

The **type** box next to the timeframe changes how the price is drawn:

* **Standard**: candles, or a line with the **Line** toggle.
* **Heikin-Ashi**: smoothed candles (close = average of open/high/low/close, open = midpoint of the previous
  candle), so a trend shows as a run of same-coloured candles. **Line** draws their close.
* **Renko**: bricks of a fixed size (the ATR over 14 candles) that only form when the close moves a full brick; a
  reversal needs two bricks of movement. Time is not an axis in Renko, so each brick sits at the time of the candle
  that completed it and several bricks from one candle share its width. **Line** is ignored.
* **Compare**: type symbols in the **vs** box (`ETH/USDT, SOL/USDT`) and press Enter. The charted symbol and those
  are drawn as percent change from the first time they all have, so everything starts at 0 %. Overlay indicators and
  signal markers are hidden (they are prices), indicator panes still show.
* **Ratio**: the charted symbol divided by the first symbol in **vs** (`KO` and `PEP` chart `KO:PEP`, see
  [Pairs](pairs.md)). Indicators and signals then run on the ratio, since that is what you are looking at.

Heikin-Ashi and Renko are only drawings: indicators, signals and backtests keep using the real candles. The
crosshair readout shows the real O/H/L/C. The type (except Compare and Ratio) is remembered between sessions.

In the terminal: `tradebot chart SYMBOL --type heikin-ashi|renko [--brick 50]`, and
`tradebot chart BTC/USDT --compare ETH/USDT --compare SOL/USDT` for the percent-change comparison. In the terminal,
Renko shows bricks only (no indicators, signals or volume).

## Indicators

**Indicators ▾** lists every registered indicator: the built-ins and [your plugins](extending.md). Each one is drawn on
the price pane (an overlay, such as moving averages or Bollinger bands) or in its own pane (RSI, MACD, ...). Pick an
indicator to add it: a dialog asks for its parameters (defaults filled in). The menu then shows `✓ rsi(14) (remove)` entries to remove it and a
**line colours...** entry to change how its outputs are drawn (colour, width, dash). The crosshair readout uses the
same colours. Colours you choose are remembered per indicator.

The dialog's **timeframe** box computes the indicator on longer candles than the chart's, e.g. the daily SMA on an
hourly chart; each candle sees the last closed longer candle only, so the line steps
([multi-timeframe indicators](multi-timeframe.md)). From the command line: `tradebot chart BTC/USDT --tf 1h -i
sma:period=50,tf=1d`.

Built-in indicators: SMA, EMA, `ma_cross`, RSI, MACD, Bollinger bands, ATR, Stochastic, Donchian, Ichimoku, Elliott
waves, Bitcoin halving cycle, ADX, Supertrend, Parabolic SAR, Keltner, CCI, Williams %R, ROC, momentum, pivots,
volume profile, anchored VWAP, Volume, OBV, VWAP, MFI, volume ratio and [pattern detection](patterns.md) (`candles`, `chart_patterns`, `support_resistance`, `volume_levels`). `tradebot list indicators` prints the full
list of your installation.

## Signals

The signal box (**No signals** by default) overlays the buy and sell markers of any strategy and the indicators
that strategy is built on (`chart_indicators()`). The `vote`, `pool` and `combo` strategies show their members'
indicators. Choosing a strategy opens a dialog for its parameters (defaults filled in). Tune them with the optimizer in the
[Backtest lab](backtest-lab.md) and press **Show on chart** there to bring a result here.

## Drawings

Toggle a tool, then click on the chart (on a touchscreen: tap):

| Tool | Clicks | Draws |
|---|---|---|
| ╱ Trend | two points | trendline |
| ― Level | one price | support/resistance line |
| ▭ Zone | two corners | rectangle |
| 𝔽 Fib | the 0 and the 1 point | Fibonacci retracement/extension levels (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.272, 1.618) with their prices |
| │ Time | one point in time | vertical line |
| 𝐓 Note | one point, then type | text note, drag to move |
| ↕ Measure | two points | price change, percent change and time between them |

Drawings are saved per symbol and come back when you load the symbol again. Right-click a drawing to **change its
colour**, **edit its text** (notes) or **delete** it; **✕ Clear** removes all drawings of the symbol. The right-click
menu on empty space also has *Reset zoom*, *Add level at <price>*, the drawing tools, *Add to watchlist* and the
candles/line and volume toggles.

## Alerts

Right-click a level and choose *Alert me at ...* to create a price alert; the sidebar's **Alerts** button lists them. See
[Alerts](alerts.md). The `regime` indicator shades the background by market regime ([regimes](regimes-and-sizing.md#market-regimes)).

## Loadouts

**Loadouts ▾** (top right) saves the current indicators and signal under a name you choose (**Save Current**) and
applies a saved set in one click; the ✕ next to a name deletes it after asking. Indicators or strategies that no longer
exist (a removed plugin) are skipped and reported. Loadouts live in `settings.json`.

## Sidebar: watchlists and news

A panel at the right of the chart with one page per category. The topmost button collapses it to a narrow strip (`»`)
and expands it again (`«`); the category buttons below it switch pages and also open a collapsed sidebar. The app
remembers whether it was open and which category was shown.

* **📰 News**: headlines for the symbol on the chart, for stocks and crypto pairs alike (Yahoo Finance). They follow
  the chart when you switch symbols and refresh every 5 minutes, and load only while the sidebar shows them. Click a
  headline to open it in the browser. The [News page](news.md) shows the headlines of all your watchlists.
* **Order book depth**: the live order book of a crypto pair as a depth chart, with the spread and which side is
  heavier; see [Crypto market data](market-data.md). Funding rate, open interest, Fear & Greed and on-chain metrics
  are indicators (panes under the chart) described there too.
* **ⓘ Fundamentals**: P/E, margins, growth, dividend yield and the earnings, split and ex-dividend dates of a stock.
  The **Events** toolbar button marks those dates on the candles (E, S, D) and, above the candles, the high-impact
  economic events (R rate decision, C CPI, J jobs data, ! other). See [Fundamentals and events](fundamentals.md) and
  [News sentiment and the economic calendar](sentiment-calendar.md).
* **⭐ Watchlists**: one or more named lists of symbols, each with its full name, last price and the day's change.
  Pick a list in the box at the top; **＋** creates one, **✎** renames it, **🗑** deletes it (after asking; the last
  list stays). Add symbols with the field (same suggestions as the chart's), with **Add current symbol**, or from the
  chart's right-click menu. Click a row to chart it; the charted symbol is highlighted. Remove a row with its ✕ (shown
  on hover) or the Delete key; reorder by drag and drop. Right-click a row (long press on a touchscreen) for Move
  up/down and Remove. Prices come from the cache first, then refresh every minute while the list is shown. The lists
  are saved in `config.toml` (`[watchlists]`), the shown list in `settings.json`.

## Touchscreens

| Gesture | Action |
|---|---|
| Pinch / spread | Zoom around your fingers. A horizontal pinch zooms the time axis, a vertical pinch zooms the price axis. |
| One- or two-finger drag | Pan |
| Flick | Pan with momentum |
| Tap | Crosshair and OHLC readout; also places drawing points |
| Double-tap | Reset the view |
| Long press | Context menu (also on right-click): reset zoom, add a level at the touched price, drawing tools, candles/line |

Lists and tables scroll with a kinetic flick. With a touchscreen present, controls automatically get finger-sized hit
targets; override this with `touch_mode = "on"` or `"off"` in `config.toml`.
