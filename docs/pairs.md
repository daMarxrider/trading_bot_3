# Pairs

The **⚖ Pairs** page analyses two assets that tend to move together: KO and PEP, two exchange-listed oil companies,
BTC and ETH. For the idea behind it see [Pairs trading](pairs-trading.md); the terminal twins are `tradebot pair`,
`tradebot pair-backtest` and `tradebot pair-scan`.

## Using the page

1. Enter the two symbols **A** and **B**, the **timeframe**, the number of **candles** (750 by default), the **window** (60 candles by default)
   and the **hedge ratio** method.
2. **Analyse**. The top plot shows both assets normalised to 100 at the start. The bottom plot shows the gap as a
   z-score with lines at 0, +2 and -2.
3. The summary line under the toolbar reports:
   * **correlation** of the two return series;
   * the **hedge ratio β** (how many units of B offset one of A, from a regression of the log prices);
   * **gap now**, in standard deviations;
   * the **half-life** in candles (how long a gap takes to close halfway; "does not revert" when it doesn't);
   * the **Engle-Granger cointegration** t value with an approximate p-value (a tick when it passes);
   * a red warning when **the pair may have stopped working**, with the reasons.

**Window** is the number of candles for the rolling hedge ratio and z-score: short follows the pair but absorbs the
gap, long reacts slowly to a changed relationship. **Hedge ratio** is *rolling* (OLS over the window) or *kalman* (an
adaptive filter without a window; the hedge ratio follows the pair smoothly).

Prices in different currencies are converted into A's currency with a Yahoo FX series (`EURUSD=X`). On daily and
longer candles the last rate at or before each candle applies. On intraday candles the last FX candle at or before
each bar applies, but only if it is at most two timeframes old; bars without a fresh rate (weekends, FX gaps) are
dropped instead of being priced with a stale one. FX intraday history has Yahoo's usual limits (1m: 7 days, 5m to
30m: 60 days, 1h/4h: 730 days). In code the age limit is `MarketDataEngine.max_fx_age`.

## `A:B` as a symbol

`KO:PEP` works as a symbol everywhere (Chart, Backtest lab, CLI): its candles are the **ratio** of the two assets,
converted into one currency. Open it in the [Chart](chart.md) to see the gap as candles, or backtest a normal strategy
on it.

## Command line

```sh
tradebot pair KO:PEP                                    # gap z-score, hedge ratio, half-life, cointegration
tradebot pair-backtest KO:PEP --window 60 --entry 2     # long one leg, short the other, fees and borrow cost
tradebot pair-backtest KO:PEP --method kalman           # Kalman-filter hedge ratio instead of a rolling window
tradebot pair-scan KO PEP PG MDLZ                       # find pairs: picked on 60% of the history, confirmed on the rest
tradebot pair-scan KO PEP PG MDLZ --same-sector --correction bonferroni
tradebot bot pair-run BTC/USDT:USDT ETH/USDT:USDT --once   # live pair bot (dry run; --live to trade)
```

See [Pairs trading](pairs-trading.md) for what the options mean and [CLI reference](cli-reference.md) for all of them.
