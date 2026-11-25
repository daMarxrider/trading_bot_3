# Data providers, timeframes and the cache

## Candles

Indicators, charts and backtests use **open, high, low, close** and, since #116, **volume** (`o.volume` in
indicators, scripts and algorithms; NaN when a symbol has none, as many indices). Prices are the same whichever
venue delivered the candles, **volume is not**: one exchange's BTC volume isn't another's, so results that use
volume can differ between data providers.

## Timeframes

`1m 5m 15m 30m 1h 2h 4h 1d 1w 1mo`. The Chart, the Backtest lab and Evolve offer all of them; the
[Pairs](pairs.md) page too. Each CLI
command lists the timeframes it takes in `--help`.

`--bars N` returns exactly N candles, also for stocks, whose markets close on weekends and overnight.
`--start`/`--end` select a date range instead. In the GUI, the chart, the Backtest lab and Evolve have optional
From/To dates that override the candle count.

Yahoo serves only limited intraday history (1m: 7 days, 5m to 30m: 60 days, 1h/4h: 730 days); a request reaching
further back returns what exists and prints a warning.

## Providers and fallback

Providers are tried in this order, and the first that has the data wins:

1. ccxt exchanges: Binance, KuCoin, Kraken (public market data, no key needed)
2. Yahoo Finance (`yfinance`): stocks, ETFs, indices, currencies
3. Alpha Vantage (needs `ALPHAVANTAGE_API_KEY`) and Financial Modeling Prep (needs `FMP_API_KEY`), skipped without
   a key

When a provider covers only part of the requested range, only the missing head or tail is requested from the next
one. Because prices are assumed identical across venues, the pieces are stitched together: Yahoo history fills in
before a coin was listed on Binance, for example.

Prices of different currencies are converted into one currency where it matters (the pair symbols `A:B`, the
portfolio), (intraday: only with a fresh exchange rate, see [Pairs](pairs.md)). Stablecoins count as the dollar.

Market data comes from third-party services under their own terms; Yahoo Finance data, for example, is meant for
personal use. See [Disclaimer and licence](legal.md).

## The cache

Candles and symbol lists are kept on disk, so the app and the CLI start without waiting for the network.

* **Candles**: one file per symbol and timeframe. The chart shows the cached candles right away, then fetches only
  the newest ones in the background. The last two cached candles are always fetched again, because the newest may not
  have been complete yet.
* **Adjusted history**: if a fresh candle differs from the cached one by more than 0.5 %, the history was split- or
  dividend-adjusted, and the series is downloaded again in full.
* **Offline**: the cached candles are shown, with a note that they may be outdated.
* **Symbol search**: crypto pairs are kept for 24 h; an older list still answers right away while a fresh one loads.
  Yahoo results are kept per query for 7 days (at most 5,000 queries). Instrument names are kept for 30 days. The
  CLI's tab completion and the app share these files.
* **Cleanup**: once a day at most, in the background. Series not used for 60 days are deleted, and 1m to 30m candles
  after 14 days. Over the size cap, the least recently used series go first, until the cache is at 80 % of the cap.
  A series keeps at most 50,000 candles.

The cache lives in the OS cache folder (see [Reference](reference.md#file-locations)) and the whole folder is safe to
delete. Configure it in `config.toml` or under [Settings](settings.md):

```toml
[cache]
enabled = true    # false: keep candles in memory only
max_mb = 500
```

```sh
tradebot cache info                 # cache folder, size, number of cached series
tradebot cache clear [--ohlc|--symbols]
```
