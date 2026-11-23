# Screener

The **🔎 Screener** page filters a list of symbols by technical values, volume and market cap. It looks at the
latest daily candle of each symbol (about 420 days of history are loaded). The terminal twin is `tradebot screen`.

To run **one indicator** over a list or an index and filter on that indicator's own outputs and alerts, use the
[Scanner](scanner.md) instead.

## Using the page

1. **Universe**: the symbols to screen, separated by commas or new lines. It starts with your watchlist symbols.
2. **Preset**: fills the filter field. *Custom* leaves it alone. Presets: oversold (`rsi < 30`), overbought
   (`rsi > 70`), uptrend (`close > sma50; sma50 > sma200`), golden zone pullback (`close > sma200; rsi < 45`),
   momentum leaders (`change_1m > 0.10`), large caps (`market_cap > 1e11`), high volatility (`atr_pct > 0.04`),
   unusual volume (`volume_ratio > 2`).
3. **Filters**: expressions separated by `;`. All must match. Write them in terms of the columns below, for example
   `rsi < 35; close > sma200; market_cap > 1e10`.
4. **Run screen**. The status bar reports `screener: N of M match`. **Double-click a row** to open it in the
   [Chart](chart.md).

## Columns

`close`, `change_1d`, `change_1m` (21 candles), `change_1y`, `rsi` (14), `sma50`, `sma200`, `atr_pct` (ATR(14) as a
share of the close), `avg_volume`, `volume_ratio` (last candle's volume over its 20-candle average; NaN without
volume data) and `market_cap` (from Yahoo; NaN when unknown).

For stocks there are also `pe`, `forward_pe`, `ps`, `profit_margin`, `operating_margin`, `revenue_growth`,
`earnings_growth`, `dividend_yield` and `days_to_earnings`. They are added when a filter uses one (or with
`tradebot screen --fundamentals`); see [Fundamentals and events](fundamentals.md).

Filters are evaluated with pandas' `query`, so any comparison and arithmetic on these columns works. From code,
`run_screen(..., extra={"stoch_k": ("stoch", {"period": 14}, "k")})` adds any indicator output as a column.

## Command line

```sh
tradebot screen -f "rsi < 35" -f "close > sma200"      # symbols default to all watchlists
tradebot screen AAPL MSFT NVDA -f "volume_ratio > 2" --sort change_1m
tradebot screen -w Crypto -f "change_1m > 0.1"         # one watchlist
```

`--sort COLUMN` sorts descending (default `change_1d`); `-f` is repeatable.
