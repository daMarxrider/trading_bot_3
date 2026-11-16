# Metrics

Every number in the Backtest lab and in the CLI results has a tooltip in the app (hover, or press and hold on a
touchscreen). The text comes from one place, `src/tradebot/glossary.py`, and this page summarises it.

## Backtest metrics

| Metric | Meaning |
|---|---|
| `total_return` | How much the money grew (or shrank) over the whole test. +50 % means 1,000 became 1,500. |
| `cagr` | The total return as an average per year (compound annual growth rate), so tests of different lengths can be compared. |
| `sharpe` | Return compared with how bumpy the ride was: the average return divided by how much it swings, per year. Rough guide: above 1 is good, above 2 is very good, below 0 lost money. |
| `sortino` | Like Sharpe, but only the drops count as risk. Higher is better. |
| `max_drawdown` | The biggest fall from a high point to a later low: how far down you would have been at the worst moment. |
| `trades` | How many trades were completed. With only a handful, the result says more about luck than about the strategy. |
| `win_rate` | The share of trades that made money. A low win rate can still be profitable if the wins are much bigger than the losses. |
| `profit_factor` | Money won divided by money lost, over all trades. Above 1 is profitable overall. |
| `exposure` | The share of the time the money was in the market. |
| `xirr` | The yearly return on the money while it was actually invested (money-weighted). It makes strategies comparable with the monthly-investing baseline, where money goes in gradually. |
| `vs buy & hold`, `vs monthly investing` | How much more per year the strategy made than that baseline, by XIRR. Negative means simply holding would have been better. |

## Report metrics (Backtest lab, `--report`)

| Metric | Meaning |
|---|---|
| `expectancy` | What an average trade earned (or lost), as a share of the money. |
| `avg_win`, `avg_loss` | The average gain of the winning trades, the average loss of the losing ones. |
| `longest_losing_streak` | The most trades in a row that lost money. |
| `time_under_water` | The share of the time the money was below its previous high. |
| `longest_under_water_days` | The longest time, in days, until the money got back to a previous high. |
| `avg_holding_days` | How long a trade lasted on average. |
| Underwater plot | How far below its previous high the money was at each moment. |
| Monthly heatmap | Return per calendar month (rows are years, last column the whole year). |

## Baselines

A strategy has to beat two simple alternatives, run on the same candles with the same fee:

* **buy & hold**: everything invested on the first candle and held;
* **monthly investing**: a fixed amount on the first candle of every month, each purchase paying the fee
  (`--dca-amount` or *Monthly amount* sets it; by default the capital is split evenly over the months).

Because money that goes in gradually makes total return and CAGR misleading, every column also shows the XIRR, and the
strategy column shows its lead over each baseline. When the strategy doesn't beat a baseline (out-of-sample, for
optimizations), the output says so in red.

## Optimizer objectives

What "best" means to the optimizer (`--objective`, *Objective*):

* `sharpe` (default): best return for the risk taken; prefers steady gains over wild swings.
* `calmar`: best yearly return compared with the worst fall from a peak; prefers small drawdowns.
* `return`: highest profit, ignoring risk; tends to pick risky settings that fit the past by luck.

## Evolver and robustness scores

* **Deflated Sharpe ratio**: the Sharpe ratio corrected for how many different strategies were tried: the chance that
  it beats the best one luck alone would produce. 0.95 or more is significant. (See [Evolve](evolve.md).)
* **Permutation p-value**: the share of runs on shuffled prices (the real run included) that scored at least as well on
  the test part. A large p means the evolver finds as much in noise.
* **Monte Carlo band** (5 % to 95 %): see [How the optimizers work](optimizers.md#monte-carlo).
