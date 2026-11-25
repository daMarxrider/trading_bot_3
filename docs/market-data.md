# Crypto market data

Besides candles, tradebot can pull a few free, keyless crypto data sets and use them like any other indicator:
on the chart (a pane under the candles), in combo rules, in [evolve](evolve.md) and in backtests.

| Indicator | Data | Source | Unit |
|---|---|---|---|
| `funding` | funding rate of the perpetual future, per settlement (positive: longs pay shorts) | Binance USD-M futures via ccxt | percent (`0.01` = 0.01 %) |
| `open_interest` | open interest of the perpetual, in the quote currency | Binance USD-M futures via ccxt | quote currency |
| `oi_change` (`period`) | change of the open interest over `period` candles | same | percent |
| `fear_greed` | Crypto Fear & Greed index, 0 extreme fear to 100 extreme greed, daily | alternative.me | 0-100 |
| `mvrv` | market value to realized value of the coin, daily | Coin Metrics community API | ratio |
| `active_addresses` | addresses active on the coin's chain per day | Coin Metrics community API | count |
| `news_sentiment` | tone of the symbol's news headlines per hour, -1 to +1; **any symbol**, not only crypto | the news feeds, see [News sentiment](sentiment-calendar.md) | score |

Every one has a `smooth` parameter (candles averaged, 1 = raw), except `oi_change`.

On-chain metrics beyond `mvrv` and `active_addresses` (exchange flows, realized cap and so on) have no free source
that needs no key, so they are not offered. The on-chain data covers the major coins only (Coin Metrics' community
coverage); another coin simply reports "no data".

## Look-ahead

A candle sees only what was known when it **opened**: the last value stamped at or before its open time. A
funding rate counts from its settlement time, an open interest value from the end of its bucket, the Fear & Greed
reading from 00:00 UTC of its day, and a Coin Metrics daily value only from the next day (it is complete when its
day is over). A value older than a few days (two for funding and open interest, three for Fear & Greed, four for
on-chain) is dropped rather than carried forward for ever, so a stale or ended series shows NaN. Because of that, an
indicator on bar `t` never changes when later candles arrive, which is what the strategy evolver requires.

## Using it

* **Chart**: add the indicator from the Indicators menu. The pane says "loading market data ..." while the data is
  fetched in the background, and "no data: ..." with the reason when it can't be had (a stock, a pair without a
  perpetual, offline without a cache). It works on crypto pairs such as `BTC/USDT` only.
* **Order book depth**: the *Order book depth* page of the chart sidebar draws the live book as a depth chart
  (cumulative amount of bids on the left and asks on the right, the mid price dotted), with the spread and how heavy
  one side is within the chosen range (1 / 2 / 5 / 10 % of the mid price; a triangle and a word say which side).
  It loads only while the page is open and refreshes every 15 seconds. It is a snapshot of Binance spot.
* **Rules**: `fear_greed < 25 & close > sma(200)`, `funding > 0.05 | oi_change(24) > 10`,
  `mvrv < 1`. `tradebot backtest` and `chart` fetch the data the strategy's indicators read automatically; the Lab
  does the same.
* **Evolve**: `tradebot evolve BTC/USDT --market-data` (or tick *Crypto market data* in the Evolve page) lets the rules
  use all of them; naming them in `--indicators` does the same. They are left out by default because they are NaN
  on stocks. A rule using missing data never fires. An evolved strategy saved as a plugin needs the data attached
  when it runs, as `backtest` and `chart` do for the indicators they can see.
* **Command line**: `tradebot market BTC/USDT [--depth]` prints the latest value of each data set and, with
  `--depth`, the order book within 0.5 / 1 / 2 / 5 % of the mid price.

Limits: market data indicators run on the candles' own timeframe only (no `tf=`), are not in the Scanner, and the
optimizer's rolling-window mode does not see them (it gets NaN); use evolve or a plain backtest for rules built on
them.

## Caching and failures

The series are cached next to the candles (`market/<data>/<symbol>.npz` in the cache folder, removed by *Clear cache*),
so history accumulates beyond what a source serves (Binance keeps only 30 days of open interest), a chart works
offline and a refresh asks for the missing tail only. Data fetched in the last five minutes is not asked for again.
Offline or rate limited, the cached series is used (and the status says so); with nothing cached the indicator shows
the reason and everything else keeps working.

For a plugin author: a `BaseIndicator` declares `market_data = ("funding_rate",)` and reads it with
`o.data("funding_rate")` (all NaN when the candles come without it). The columns are `funding_rate`,
`open_interest`, `fear_greed`, `mvrv` and `active_addresses`; `tradebot.data.market.MarketData().attach(df, symbol)`
adds them to a candle frame.

None of this is investment advice: a sentiment or leverage reading is a heuristic, check any rule in a backtest.
