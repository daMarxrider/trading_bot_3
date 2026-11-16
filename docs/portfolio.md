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

## Analytics

Below the table, **Analyze** (and `tradebot portfolio-analytics`) adds four views. Load the portfolio first.

| View | What it shows |
|---|---|
| Performance | Value over time rebuilt from the [journal](journal.md)'s live and paper fills, with the **time-weighted return** (TWR, independent of when money went in, the figure to compare with a benchmark), the **money-weighted return** (MWR, the annualised internal rate of return of your deposits and withdrawals) and the same period of a **benchmark** (`SPY` by default; `URTH` for MSCI World, `BTC/USDT` for Bitcoin) |
| Correlation | Correlation matrix of the daily returns of the current holdings (last year, days on which all of them trade) |
| Risk | Per holding: weight, annualised volatility and **share of risk**, the part of the portfolio's variance it causes. A share well above the weight marks a volatile or highly correlated holding |
| Rebalancing | Fill in a **target allocation** (`AAPL=40, MSFT=40, BTC/USDT=20`, percent, any scale) to see what to buy or sell to get back to it. Sells fund the buys, so the amounts net to zero. A holding without a target is sold |

The history only knows orders recorded in the journal (bots run with the journal on); holdings bought elsewhere appear in
correlation, risk and rebalancing but not in the performance curve. A buy counts as money paid in (plus fee), a sell as
money taken out. Crypto holdings are named `BTC/USDT`, stocks by ticker, in the targets too. Rebalancing is a
suggestion only: nothing is ordered.

```sh
tradebot portfolio-analytics --broker scalable --benchmark URTH
tradebot portfolio-analytics --target "AAPL=50,MSFT=30,BTC/USDT=20" --tolerance 1
```

`--tolerance` leaves holdings within that many percentage points of their target alone; `--mode` picks which journal
fills to use (default `real`: live and paper).
