# How the optimizers work

`tradebot optimize` and the Backtest lab's **Optimize** button search a strategy's parameters (the ranges are the
`Param`/`IntParam` limits each strategy declares, see [Extending](extending.md)). Everything here applies to both;
the options are in the [CLI reference](cli-reference.md#tradebot-optimize), the form in the
[Backtest lab](backtest-lab.md). For strategies written from scratch see [Evolve](evolve.md).

## One execution model

A strategy returns a **target position per candle** (1 long, 0 flat, -1 short when it allows shorts). The position
decided at candle *t*'s close earns candle *t+1*'s return, and fees (plus optional slippage) are charged on every
position change. The backtester, the optimizer and the live bot all use this model, so there is no look-ahead and live
behaviour matches backtests.

## Train and test

`optimize` tunes on the first 70 % of the data (`--train`) and reports the remaining 30 % **out-of-sample**. Candidates
with fewer than `--min-trades` round trips (default 4) are rejected. Only the out-of-sample numbers say whether the
parameters generalise; a warning flags overfitting when the in-sample winner does worse out of sample.

## GA, PSO, or both

`--method` chooses the search:

* `ga`: genetic algorithm, the default: tries many parameter sets, keeps the best, mixes and slightly changes them.
* `pso`: particle swarm: a swarm of guesses drifting towards the best settings found by themselves and by the swarm.
* `both`: runs both one after the other on the same data, split, budget and seed, and reports how much they disagree
  (difference per parameter as a share of its range, the in-sample score gap, both out-of-sample results).

The comparison ends with a verdict: `consistent` (both found the same region, the optimum looks well-defined),
`fragile` (different parameters, nearly the same score: the exact values are mostly noise) or `divergent` (one method
got stuck in a weaker region).

## Backends and speed

The optimizer evaluates a whole **population** per call. `tradebot device` shows the backend: CUDA, ROCm or a
multi-threaded NumPy pool (`--backend auto|gpu|cuda|rocm|cpu|numpy`). Strategies written with
[`ops` only](indicators-as-algorithms.md#staying-vectorized) are batched; strategies with `vectorized = False` (such as
`elliott_wave`) are evaluated candidate by candidate on the CPU. Integer parameters are always plain `int`; the
optimizer groups candidates by them, except for `period_batched` strategies (`macd_cross`, `sma_cross`), which batch
different window lengths in one call. Setup: [Getting started](getting-started.md#gpu-acceleration).

## Several symbols

Parameters tuned on one chart are often curve-fitted. Give `optimize` several symbols (comma-separated, or
`--watchlist`/`-w`, repeatable; `all` = every watchlist) and one parameter set is scored on every symbol, each with its
own 70/30 split. `--aggregate` combines the per-symbol scores: `mean` (default), `median` (one outlier can't decide),
`min` (the worst symbol counts: the most cautious) or `mean-std` (mean minus the spread between symbols).

A symbol with less than half the candles of the longest one is skipped with a note; `--align` instead cuts every
symbol to the dates all of them cover. The output has a row per symbol, the aggregate, on how many symbols each
baseline was beaten, and a warning when one or two symbols carry the result.

Two optional extras, off by default:

* `--min-symbols-beating N`: parameter sets whose in-sample objective doesn't beat buy & hold on at least N symbols
  are rejected, like those with too few trades.
* `--samples K --sample-length L` (**random windows**): each generation scores the candidates on K random stretches of
  L candles per symbol instead of all in-sample candles, each with its own random start (plus warm-up candles for the
  indicators). All windows have the same length, so a vectorized strategy computes
  `[windows x candidates x candles]` in one batch, which suits the GPU. Settings that only fit one period lose. The
  progress score is noisy rather than rising steadily, and the winner is chosen from the final population on all
  in-sample candles.

## Walk-forward validation

One 70/30 split is a single draw. `--walk-forward N` runs the optimizer N times: each run (a *fold*) tunes on a
training window and is scored on the next, unseen stretch; then everything moves forward:

```
|-- train 1 --|- test 1 -|
     |-- train 2 --|- test 2 -|
          |-- train 3 --|- test 3 -|
```

* `--wf-train` / `--wf-test`: candles per training window and per test stretch (default: the folds fill the data and
  keep the `--train` ratio).
* `--anchored`: every training window starts at the first candle and grows.
* `--holdout N`: the last N candles stay out of every fold and are scored once at the end, with parameters tuned on the
  stretch right before them.

It works with `--method ga|pso|both` (with `both`, each fold keeps the in-sample winner), several symbols, random
windows and `--min-symbols-beating`. The output has a row per fold, the combined out-of-sample score (mean and spread),
in how many folds each baseline was beaten, the tuned parameters per fold and how far each one moves (as a share of its
range). Red warnings flag a score that swings more between folds than its average, a strategy that beats buy & hold in
fewer than half the folds, parameters that move across more than half their range, and a holdout that doesn't beat
buy & hold. In the lab, walk-forward is **on by default** (5 rolling folds, automatic windows); the holdout is off.

## Meta-search

Which optimizer settings give the best strategy (random windows or not, their length, `--min-symbols-beating`) is
otherwise trial and error. `--meta N` tries N combinations of them (random picks; your own settings first), runs a
walk-forward (`--walk-forward` folds, default 5) with each and ranks them by the mean out-of-sample score over the
folds. `--meta-compute` also tunes population and generations, `--meta-time M` stops starting new combinations after
M minutes, and Ctrl+C ends the search early (the holdout is still scored). The objective and `--aggregate` stay your
choice: they define what "better" means.

Every extra tuned setting is another chance to fit noise, so the data has three levels: the optimizer tunes on each
fold's training part, the combinations are ranked on the folds' test parts, and a holdout nothing has seen (`--holdout`,
by default as long as one fold's test stretch) is scored once with the winner. **Only that last number is the result.**
Each combination costs a full optimization per fold; the command prints the number of optimizer runs first. In the lab,
*Use the winning settings* copies the winner into the Optimizer card.

## Picking the strategies (the pool)

Without an algorithm, `optimize` also decides *which* strategies and indicators to use. It starts from a pool: every
strategy and indicator (plugins included), or the ones given with `--pool`. Every member gets an on/off switch next to
its parameters, and the search tunes both.

* Active strategies vote, as in `vote`, with an optimized threshold.
* An indicator can't trade on its own, so it joins as a **filter** with a default rule: a buy also needs every active
  filter to agree: `sma`/`ema`: close above the average; `rsi`/`stoch`: oversold; `macd`: MACD above zero;
  `bbands`/`donchian`: breakout; `elliott`: wave 3; `halving`: a window of days after a Bitcoin halving. Indicators
  without a rule (`atr`, `ichimoku`, `ssw`, and plugins) are left out, with a note.
* `--min-members` / `--max-members` limit how many members are used; at least one strategy always is. Each member costs
  a small fitness penalty (0.01), so of two equally good combinations the smaller one wins.

The output lists each member as used or dropped, the parameters of the used ones and the command to reuse the result
(`tradebot backtest ... pool --members ...`). With `--method both`, the table also shows whether GA and PSO picked the
same members: agreement is a good sign that the selection is real. Strategies set `poolable = False` to stay out
(`range`, `vote`, `combo`, `pool`).

## Slippage

Fills at the close are optimistic, most of all on short timeframes and thin symbols. `--slippage 5` (basis points per
side, 0.05 %) charges a cost on every position change on top of the fee. It is 0 by default, so existing results do
not change, and it is applied identically in `backtest`, `optimize` (GA/PSO, NumPy and torch), `evolve`, walk-forward
and the baselines. `--slippage auto` uses a suggestion for the asset class (crypto 5, stocks 2, currencies 1 bp). Set a
default in `config.toml` (see [Reference](reference.md#configtoml)); the Backtest lab and Evolve have a "Slippage / side"
field initialised from it.

## Monte Carlo

A backtest is one ordering of the trades. `tradebot montecarlo` (and the lab's Monte Carlo card) re-draws the trades
or the candle returns many times to show how much of the final return and the worst drawdown is luck in the order:

* `shuffle`: the same trades in random order (the final return is the same, only the drawdown moves);
* `bootstrap` (default): draw with replacement, so the final return varies too;
* `block`: bootstrap in runs of consecutive items, which keeps streaks together.

The 5 % to 95 % band is reported, plus the probability of a drawdown above `--max-dd` (default 20 %). `--source`
picks trades or candle returns, `--seed` makes it repeatable.
