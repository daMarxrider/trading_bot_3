# Backtest lab

The **🧪 Backtest** page tests a strategy on history, optimizes its parameters, checks the result with walk-forward and
Monte Carlo, and compares it with simple baselines. The terminal twins are `tradebot backtest`, `optimize` and
`montecarlo`. Read [Metrics](metrics.md) and [How the optimizers work](optimizers.md) for what the numbers and options
mean; every field also has a tooltip.

The settings are a column on the left, the results on the right. Drag the dividers between the result panes and between
the settings and the results; tall cards scroll inside their pane. The sizes are remembered, and double-clicking a
divider resets them.

## 1. Choose what to test

* **Symbols**: one or several, comma-separated (completion works on the last one). The **☰** button fills in a watchlist
  or all of them.
* **Timeframe**, **Candles** (1,500 by default) or **Dates** (From/To).
* **Fee / side** (0.001 = 0.1 %), **Slippage / side** in basis points (initialised from `[backtest] slippage_bps`),
  **Monthly amount** of the monthly-investing baseline (auto: capital split over the months).
* **Algorithm**: any strategy ([Strategies](strategies.md)), plugins and saved combos included. Depending on the
  strategy one more card appears:
  * `vote`: **Members**, the strategies that vote (*All* or a selection);
  * `pool`: **Pool**, the candidates the optimizer picks from (strategies and indicator *filters*) with **Min/Max
    members**;
  * `combo`: **Combo builder**, entry and exit rules as text, filled by picking a block, a logic and a threshold, with
    an *Intrabar* switch for fills and **Save combo**.
* **Position sizing**: a layer that decides how much to put on (fixed, volatility target, fixed risk, Kelly); see
  [Market regimes and position sizing](regimes-and-sizing.md#position-sizing).
* **Parameters**: a form generated from the strategy, with the range the optimizer searches.

## 2. Run

* **Run backtest**: the strategy with the current parameters. The plot shows the equity curve against **buy & hold** and
  **monthly investing**; the table shows the metrics for all three. A red note appears when the strategy doesn't beat a
  baseline. The ↕ and **log** buttons in the plot's bottom-left corner fit the value axis and switch to a log scale.
* **Optimize**: searches the parameters with the settings of the Optimizer card, with live progress. In-sample (the
  first 70 %) and out-of-sample (the rest) are shown separately.
* **Show on chart**: opens the symbol in the [Chart](chart.md) with this strategy's signals.
* **Stop** (only during a meta-search): no new combinations are started; the best one so far is still scored on the
  holdout.

The footer of the Optimizer card shows the **compute backend** (`tradebot device`).

## The Optimizer card

| Field | Meaning |
|---|---|
| Method | Genetic algorithm, particle swarm, or both and compare them |
| Population / swarm, Generations / iterations | The size and length of the search |
| Objective | `sharpe`, `calmar` or `return` |
| Several symbols | How per-symbol scores combine: mean, median, min, mean-std |
| Align dates | Cut all symbols to their common date range |
| Beat buy && hold on | Reject parameter sets that don't beat buy & hold on at least this many symbols |
| Random windows, Window length | Score each generation on random stretches instead of all candles |
| Walk-forward | Number of folds (**on by default: 5**; *off* = one 70/30 split) |
| Train window, Test window, Anchored, Holdout | Walk-forward windows in candles (auto by default), growing training window, last candles kept out |
| Meta-search, Meta: also tune | Try combinations of the optimizer's own settings (needs walk-forward) |

## Reading the results

* **Per-symbol table** (several symbols): each symbol's score, the aggregate and the baselines beaten; click a row to
  see that symbol's curves and metrics.
* **Walk-forward**: the plot shades every fold's test stretch (numbered) and draws one *walk-forward* curve through
  them, each stretch traded with the parameters its fold tuned, what walk-forward trading would have earned; the holdout
  continues it. The **folds table** has a row per fold, the combined row and the holdout, then the warnings and the
  parameters per fold. Click a fold to see its own curves, *all* to return.
* **Meta-search card**: every combination, best first, with the winner's lead over the median, the holdout result and
  the warnings. Click a combination to see its folds. **Use the winning settings** copies them into the Optimizer card
  and turns the meta-search off.
* **Selection card** (pool): which members were used or dropped, and whether GA and PSO agree.
* **Report card**: expectancy, average win and loss, longest losing streak, time under water, average holding time, an
  **underwater plot** and a **monthly returns heatmap**, with two export buttons: **Export trades (CSV)** and
  **Export report (HTML)**, one self-contained page with the metrics, equity, underwater plot, monthly returns and trades.
  The CLI equivalents are `backtest --trades-csv FILE` and `backtest --report FILE`.
* **Monte Carlo card**: set the number of runs, the method (bootstrap trades, shuffle trades, block bootstrap of candles)
  and a drawdown threshold, then **Run Monte Carlo**. The 5 % to 95 % band of final return and drawdown and the fan of
  curves show how much of the result is luck in the order of the trades.

* **Regime breakdown**: the report card shows return, Sharpe, exposure and trades per market regime
  ([regimes](regimes-and-sizing.md#market-regimes)).
* **Several symbols on one account**: the **Basket** page, see [Basket](portfolio-backtest.md).

## Fills

Stops, take-profits and limit entries of a **combo** can be filled at the close (default) or *Intrabar*, inside the candle
against its high and low. All other strategies trade at the close. Details in [Strategies](strategies.md#the-combo-builder).

## Command line

```sh
tradebot backtest AAPL macd_cross --bars 2000 -p fast=8 -p slow=21
tradebot backtest AAPL,MSFT,NVDA macd_cross -p fast=8          # one row per symbol
tradebot backtest AAPL macd_cross --report aapl.html --trades-csv aapl.csv
tradebot optimize BTC/USDT rsi_threshold --population 1024 --generations 40 --objective calmar
tradebot optimize AAPL macd_cross --walk-forward 5 --holdout 250
tradebot optimize -w Crypto sma_cross --meta 20
tradebot montecarlo AAPL macd_cross --runs 2000 --method block
```

More in [Command line](cli.md).
