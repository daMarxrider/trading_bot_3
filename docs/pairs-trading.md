# Pairs trading

Two assets that usually move together (same sector, same index, two share classes) sometimes drift apart. Pairs
trading bets that the gap closes again: go **long the cheap one and short the expensive one** at the same time. The
result barely depends on where the market goes, only on the gap. The [Pairs page](pairs.md) and the commands
`pair`, `pair-backtest`, `pair-scan` and `bot pair-run` implement it.

## The numbers

* **Hedge ratio (β)**: how much of B offsets A. Fitted on log prices, either with a rolling regression (`--method
  rolling`, over `--window` candles) or a Kalman filter (`--method kalman`).
* **z-score of the gap**: where the log spread is now, in standard deviations from its mean. Beyond ±2, the pair is
  stretched.
* **Half-life**: how many candles a gap needs to close halfway, from an AR(1) fit. A spread that doesn't come back
  has none, and is no pair.
* **Cointegration (Engle-Granger)**: regress A on B, then test whether the leftover gap wanders off forever or comes
  back. The p-value is approximate (interpolated between critical values).
* **Broken-pair warning**: flags a pair whose relationship seems to have changed.

## The trading rule

On each closed candle, the position depends on the z-score of the gap:

* open when the gap is `--entry` standard deviations wide (default 2): long A and short B if the gap is negative,
  the reverse if positive;
* close when it is back inside `--exit` (default 0.5);
* give up beyond `--stop` (default 4).

The capital of a trade is split by the hedge ratio at entry (`w_a = 1 / (1 + |β|)`, `w_b = β / (1 + |β|)`) and the
weights stay fixed until the exit. Same execution model as everywhere in tradebot: a signal from the candle closing at
*t* is filled at that close and earns the next candle's return. Costs: the fee on every change of either leg, and
`--borrow` (yearly, default 0.02 for stocks, use 0 for crypto) on the short leg for every candle it is held.

## The chance that a gap closes

A z-score of 2 means something different for a pair whose gaps close in three candles and one that takes thirty.
`pair` shows, and `--min-probability` filters on, **how likely a trade opened now reaches the exit band before the
stop within `--horizon` candles** (default 30).

* The gap's z-score is modelled as mean-reverting (an Ornstein–Uhlenbeck process): pulled back to 0 at a speed
  `κ = ln 2 / half-life`, with a long-run spread of 1. The speed comes from the last `2 × --window` z-scores (a
  speed needs more candles than a z-score), corrected for the small-sample bias of that fit; the candle itself never
  enters its own estimate.
* By the Feynman–Kac formula, the chance solves the backward equation of that process with the exit band and the
  stop as barriers. The speed and the horizon only enter as `κ × horizon`, so the equation is solved once per exit
  and stop into a table and every candle is a lookup. Without a time limit it has a closed form, which the tests
  check the table against, as well as a Monte Carlo simulation.
* `--min-probability 0.6`: open only when the chance is at least 60 %. Trades that are already open are not
  affected. The live bot logs every entry the filter kept it out of, with the chance it saw. 0 (the default)
  switches the filter off.

While the speed is unknown (fewer than `3 × --window + 2` candles, or no reversion seen in the last `2 × --window`) the
filter doesn't open. The chance is only as good as the model: a pair that breaks apart still looks mean-reverting
until the window has caught up, so keep the stop, and let `pair-backtest` (or a walk-forward test) show whether the
filter helps a pair.

## Finding pairs

`pair-scan` tests every combination of the symbols you give it. Testing hundreds of pairs finds some that look
cointegrated by chance, so it works in two steps:

1. Pairs are **picked on the first part** of the history (`--split`, default 60 %) with a pre-filter on return
   correlation (`--min-correlation`, default 0.7) and optionally the same sector (`--same-sector`).
2. They are **confirmed on the rest**, with a correction for how many pairs were tested: `fdr` (Benjamini-Hochberg,
   the default), `bonferroni` (stricter) or `none`.

`--backend gpu` batches the tests on a CUDA/ROCm GPU; `auto` picks it when there is one.

## Live

`tradebot bot pair-run LEG_A LEG_B` runs the rule live (dry run unless `--live`). The short leg is an ordinary SELL,
so the broker needs margin or futures for it: pass symbols that broker can short, for example perpetual contracts
`BTC/USDT:USDT`. A spot account rejects the sale; the bot then undoes the leg already placed and stops trading that
step.

**Remembering the open pair.** With `--live` the bot writes the open pair (side, amount in each leg) to a JSON file,
atomically, and picks it up after a restart (`$TRADEBOT_PAIR_STATE_DIR`, default the `pair_bots` folder of the config
folder; one file per broker and legs). Every trade is written ahead: recorded as *pending* before the first order and after
each fill, cleared only when the trade finished. A restart that finds a pending trade (a crash mid-trade, a failed undo)
refuses to trade until you have checked the account and run `tradebot bot pair-state LEG_A LEG_B --keep` or `--forget`.
On start the saved pair is also checked against the broker's positions where the broker can report them. A partly filled
first leg shrinks the second to keep the hedge; a partly filled close is finished at the next candle. Anything whose
outcome the bot can't know (a timeout placing an order, a failed undo) halts the bot with the pending trade kept.
See [Brokers and live trading](live-trading.md).
