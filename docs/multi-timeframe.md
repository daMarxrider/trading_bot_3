# Multi-timeframe indicators and strategies

An indicator normally sees only the candles it runs on. With `tf=` it runs on longer candles instead, built
from the ones you trade, so an hourly strategy can filter on the daily trend:

```sh
tradebot backtest BTC/USDT combo --tf 1h --entry "rsi(14) < 30 & close > sma(200, tf=1d)" --exit "rsi(14) > 70"
```

## No look-ahead

The longer candles are calendar buckets in UTC: `4h` from midnight, `1d` days, `1w` weeks from Monday,
`1mo` calendar months. Each one is aggregated from the shorter candles (first open, highest high, lowest low,
last close, summed volume).

A shorter candle only sees a longer candle once that longer candle has **closed**. A candle's close is its open
time plus the usual distance between candles. On a 1h chart the daily SMA therefore changes on the 23:00 candle,
which closes at midnight, and stays the same until the next one. Before the first closed day it is empty, like
any indicator that is still warming up. A value never changes when later candles arrive.

Markets that close before the bucket ends (a stock's last hourly candle closes at 16:00, its day at midnight)
use the day from the next morning's first candle. That is one candle later than strictly needed, never early.

## Where it works

- **Rules** (backtest, optimizer, walk-forward, bot, saved combos, the Lab's combo builder): `tf=` on any
  indicator, pool rule or strategy:
  - `close > sma(200, tf=1d)`, `rsi(14, tf=4h) < 30`, `regime(tf=1w).bull`, `rsi(tf=1d)` (the RSI pool rule
    on daily candles), `macd_cross(tf=1d)` (the strategy's position on daily candles).
  - Parameters get the timeframe in their name (`sma_1d.period`). `sma(50)` and `sma(50, tf=1d)` are separate
    members, so the optimizer tunes them like any other parameter.
  - Prices take no `tf=`. Use a one-candle average for a longer candle's close: `sma(1, tf=1d)`.
  - In random-window optimization the warm-up counts in the longer candles: `sma(200, tf=1d)` on hourly
    candles needs 4,800 candles before each window.
- **Chart**: the indicator dialog has a timeframe box (CLI: `-i sma:period=50,tf=1d`). The line steps once
  per longer candle. If the timeframe is shorter than the chart's candles, the indicator is drawn on the chart's
  own candles and its label says why. Loadouts and the remembered session keep the timeframe.
- **Scanner**: the "indicator:" box (CLI `--indicator-tf 1d`) runs the indicator on that longer timeframe and
  ignores the candle that is still running, so the row shows what a strategy using `tf=1d` sees right now.
  Opening a row charts the indicator on that timeframe.
- **Evolve**: `--higher-tf 1d,1w` (the app: "Also on" under Indicators) offers every indicator output a second
  time on those candles, e.g. `close > sma(50, tf=1d)`. The exported plugin computes them the same way. Not
  available with `--from-scratch`, which uses no ready-made indicators.

A timeframe must be longer than the candles (`tf=4h` on daily candles is an error), and the candles need
timestamps.

## For plugin authors

`tradebot.core.mtf.Higher(o, "1d")` gives the daily candles of `o` and `.map(fn)` lines any result up with
`o`'s candles. `mtf.compute(cls, o, "1d", **params)` does that for an indicator class, and
`SMA(tf="1d", period=200)` builds an indicator instance that draws on the daily candles.
