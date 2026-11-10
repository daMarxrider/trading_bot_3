# Evolve settings benchmark (#101)

`uv run python scripts/bench_evolve.py run [--methods ...] [--seeds ...]`, then `... report` (writes
[report.md](report.md); raw rows in `results.jsonl`, one JSON object per run).

**Question:** do the "Good" / "Best result" presets and Auto-configure give a better result *on data nobody
searched on* than the fast defaults?

**Setup:** 8 datasets (AAPL, SAP.DE, BTC/USDT, ETH/USDT daily; BLK weekly; a basket of MSFT, GOOG, BLK; two
random walks as a control). Score = Sharpe ratio on the newest 20 % of the candles. Every method got the same
seed per dataset; fast, good and auto-good ran with 4 seeds (7-10), best with 1 (seed 7, 1-4 hours per run).
Auto: best was not run (several times the cost of best).

## Results (real datasets, 6 × seeds)

| method | mean test Sharpe | median | share > 0 | train-test gap | run time |
|---|---|---|---|---|---|
| fast (defaults) | 0.12 | 0.16 | 15/24 | 1.92 | seconds |
| good | 0.03 | 0.07 | 12/24 | 1.77 | ~12 min |
| auto-good | 0.08 | 0.22 | 15/24 | 1.71 | ~15 min |
| best (1 seed) | -0.05 | – | 3/6 | 2.02 | ~1.7 h |

Paired over dataset and seed (24 pairs), the differences are far inside the noise: auto-good minus fast
-0.05 ± 0.18 (standard error), auto-good minus good +0.04 ± 0.17, good minus fast -0.09 ± 0.24. The scores of
one method vary by about ±0.8 between seeds on the same data.

## What this says

1. **No method is measurably better on unseen data.** Not the bigger search, not the longer search, not the
   automatic choice of settings. With this noise, an effect below about 0.35 Sharpe cannot be detected.
2. **More search overfits more, not less:** the gap between training and test score is largest for best (2.02)
   and fast (1.92), smallest for auto-good (1.71). The old assumption "more search effort is pure gain" is
   not supported, and the text in the code and glossary was changed accordingly.
3. **The checks work as intended.** On the two random walks no run reported an edge (p <= 0.05 and test > 0:
   0 of 18 runs with a permutation test). On real data only one run was significant: best on the basket
   (p = 0.05, test Sharpe 1.26), a single result among about 8 best runs, so not strong evidence.
4. **The value of the presets is the evidence they produce** (shuffled-price p-value, walk-forward folds,
   deflated Sharpe), not a better strategy. Fast results come without any way to tell luck from edge.
5. **What Auto-configure chose** (32 runs, median): depth 6, size 45, parsimony 0.009 (the presets use
   0.003-0.004), 4 segments, minimum 14 trades, aggregate "median" in 20 of 32. Higher parsimony and a robust
   aggregate are a pattern worth testing, but this benchmark cannot confirm them.

## Not covered

* Auto: best, the from-scratch mode, more than 4 seeds for best, more symbols and time frames.
* A direct test of parsimony 0.01 against the preset values (needs several seeds to see anything).
* The numeric preset values are reasoned, not tuned: this data gives no grounds to change them.
