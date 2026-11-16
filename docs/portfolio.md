# Portfolio

The **💼 Portfolio** page shows what you hold at one or more brokers: total equity, allocation donuts (by holding and
by asset class), and dividend metrics. The terminal twin is `tradebot portfolio`.

## Using the page

1. Tick the **brokers** to include (one checkbox per registered broker). The page needs their credentials or holdings
   file, see [Brokers and live trading](live-trading.md#brokers) and [Settings](settings.md).
2. **Load portfolio**.
3. Read the key figures and the table. **Double-click a holding** to open it in the [Chart](chart.md).

The portfolio's symbols also feed the symbol suggestions and the [News page](news.md).

## Figures

| Figure | Meaning |
|---|---|
| Total equity | Value of all holdings |
| Current dividend yield | Forward annual dividend divided by the price |
| Annual (trailing) yield | Dividends paid in the last 12 months divided by the price |
| Yield on cost | Forward annual dividend divided by your average purchase price |
| Annual dividend income | Expected dividends over the next year |

## Command line

```sh
tradebot portfolio --broker scalable --broker binance
tradebot positions --broker kucoin      # balances at one broker
```

Scalable Capital and Trade Republic have no trading API: their holdings come from a CSV
(`symbol,quantity,avg_cost`), set with `holdings_csv` under `[brokers.scalable]`.
