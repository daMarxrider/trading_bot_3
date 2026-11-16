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
step. The bot doesn't remember an open pair across a restart, so check the broker's positions before starting again.
See [Brokers and live trading](live-trading.md).
