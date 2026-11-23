# Evolve

`tradebot evolve` and the **🧬 Evolve** page don't tune a strategy you already have. They **write one**: the entry and
exit rules are evolved with genetic programming, including which indicators they use, how those are combined, and every
number in them. A result looks like this:

```
entry = crosses_above(ema(close, 13), sma(close, 41)) and rsi(9) < 58
exit  = close < shift(lowest(low, 7), 1) or bars_in_trade >= 30
```

```sh
tradebot evolve AAPL                                   # one symbol, default settings
tradebot evolve AAPL,MSFT,NVDA --generations 60 --population 500
tradebot evolve -w Crypto --tf 4h --indicators rsi,ema,atr --max-depth 4
tradebot evolve AAPL --permutations 5                  # also run it 5 times on shuffled prices
tradebot evolve AAPL --save my_rule                    # keep it as a plugin algorithm
```

## The page

The form (left) has three groups, each with tooltips:

* **Data**: symbols (the ☰ button fills in a watchlist), timeframe, candles or dates, fee, slippage, monthly amount;
* **Search**: population, generations, *From scratch* (see below), max depth and size, islands, objective, how several
  symbols combine, aligned dates, whether exit rules may look at the open trade, the seed;
* **Keeping it honest**: validation and test fractions, segments, minimum trades, parsimony, shuffled-price runs,
  walk-forward folds;

plus the **Indicators** the rules may use. The preset buttons set everything at once: **Fast (defaults)**, **Good**,
**Best result** and, much slower, **Auto: good** and **Auto: best** (a search picks the settings for your data first).
**Evolve** starts, **Stop** ends a run early.

The result shows the rules (with the dates of the parts and the seed), the warnings in red, the equity of the train,
validation and test parts (the latter two shaded) against buy & hold and monthly investing, the best score per
generation in training and on validation (training rising while validation stays flat means overfitting), a row per
symbol (click one to plot it), and the best strategies in training with an arrow at the one picked on validation. Both
plots have the chart's ↕ auto-fit and a log button, double-click resets the view. The panes can be resized with their
dividers (double-click a divider to reset).

**Save as plugin...** keeps the result under a name you choose; it appears in the Backtest lab's and the chart's
strategy lists right away. **Show on chart** shows its signals without saving it.

## Building blocks

The rules are trees made of:

* the candles (open, high, low, close) and every indicator output, plugins included. An output that looks into the
  future is left out automatically, like the Elliott zigzag: the evolver checks that its values don't change when later
  candles are added;
* `sma`, `ema`, `highest`, `lowest`, `shift`, `max`, `min`, `if_else`, `change(x, n)` (relative change), `gap(a, b)`
  (a/b - 1), `volatility(n)`, `price + k x atr(n)`;
* comparisons (`>`, `<`, `crosses_above`, `crosses_below`), `and`, `or`, `not`;
* in exit rules, the open trade: `bars_in_trade`, `stop_loss`, `take_profit` and `trailing_stop` (fractions of the
  price; `--no-state` turns them off);
* with `--higher-tf 1d,1w` (the page: **Also on** under Indicators), every indicator output again on those longer
  candles, e.g. `close > sma(50, tf=1d)` as a daily trend filter for hourly rules
  ([multi-timeframe indicators](multi-timeframe.md)).

Every part has a **unit**: a price, a 0-100 oscillator, a relative change, a price distance (ATR), or an indicator's own
unit like MACD. Only like is compared with like, and a price is never compared with a fixed number, because
"close > 182.5" means nothing on another symbol. Every tree the search builds is therefore valid, and it means the same
on every chart. Plugin indicators can declare their outputs' units with `units = {"value": "osc"}`; otherwise overlays
count as prices, and indicators whose `levels` lie within 0-100 count as oscillators.

## The search

It starts with 300 random programs (`--population`). In each of 40 generations (`--generations`) it keeps the best few
and breeds the rest by swapping parts between two good programs (or a whole entry or exit rule), and by mutation: a new
random part, `>` and `<` swapped, `sma` and `ema` swapped, a nudged number, or a part replaced by one of its own
sub-parts. `--max-depth` (6) and `--max-size` (40 building blocks per rule) cap the size, and every building block costs a
little fitness (`--parsimony`, 0.002): of two equally good rules the shorter one wins. Equivalent rules are recognised
(`a and b` = `b and a`, `x > x` is never true), so each is scored once.

The search runs on the CPU with NumPy; parts that many programs share (such as `rsi(14)`) are computed once and cached.
A run with the defaults on 2,500 daily candles of one symbol takes under a minute; time grows with the number of
symbols, and `--permutations N` repeats the whole search N times.

## Keeping it honest

A search this large finds patterns in pure noise, so the data is split by date, all symbols at the same dates:

```
|---------- train (3 segments) ----------|-- validation (20 %) --|-- test (20 %) --|
```

* The fitness only sees the training part. It's scored in `--segments` consecutive stretches on every symbol and combined
  with `--aggregate`, so a rule has to work in every stretch and on every symbol. Programs with fewer than `--min-trades`
  round trips on a symbol are rejected.
* The 30 best programs ever seen are compared on the validation part, and the best one there is the result. Evolution
  never sees validation scores.
* The result is scored once on the test part (`--test`), which no step of the search has seen, next to buy & hold and
  monthly investing.
* `--permutations N` runs the same search N more times on shuffled candles. The **p-value** is the share of runs (the real
  one included) that scored at least as well on the test part. A large p means the evolver finds as much in noise.

The output shows the rules, a row per symbol, the best programs with their validation scores, and warnings in red: the
result doesn't beat buy & hold on the test part, scores much worse on unseen data than in training, makes fewer than 3
test trades, depends on one or two symbols, or fails the permutation test. Every run prints its `--seed`, so it can be
repeated.

## From scratch

`--from-scratch` (the *From scratch* box): **no ready-made indicators at all.** Rules are built from the candles and
low-level maths, so the evolver can invent RSI, Stochastic, Bollinger, ATR or MACD itself, or something nobody has named
yet. Runtime doesn't matter in this mode; it trades speed for expressiveness and safeguards.

* Building blocks: `diff(x, n)`, `pos(x)` / `neg(x)` (up and down moves), `rolling_sum`, `rolling_std`, `wilder`
  (Wilder's smoothing), `sma`, `ema`, `highest`, `lowest`, `shift`, `abs`, free constants, `100 * ratio`, `squash(ratio)`
  (onto 0-100) and `+ - x /`.
* The arithmetic is checked by **dimensional analysis**, like units in physics: `price - price` is a distance,
  `price +- distance` a price, `price / price` a ratio. `price + price` and `price x price` can't be built, so a rule
  still means the same on every symbol.
* RSI(14) is `100 * wilder(pos(diff(close, 1)), 14) / (wilder(pos(diff(close, 1)), 14) + wilder(neg(diff(close, 1)), 14))`.
  The tests build all five indicators like this and check that they equal the built-in ones. A slow test hides an
  `RSI(14) < 30` rule in synthetic prices: evolve gets close but doesn't find RSI itself yet.
* **Evolved building blocks**: every 5 generations, parts that keep showing up in the best programs become named modules
  (`m1`, `m2`, ...), used as a single building block from then on. The rules print with a `where` section defining them.
* Limits are raised to depth 12 and 100 building blocks; windows and lags are searched on a log scale.
* **Islands** (`--islands`, default 4): the population is split into groups that evolve apart and pass their 2 best on to
  the next group every 10 generations.
* Safeguards on by default: 5 shuffled-price runs, 3 walk-forward folds (`--folds`), the **deflated Sharpe ratio** (warning
  below 0.95) and deduplication of strategies that trade exactly alike.

```sh
tradebot evolve BTC/USDT --from-scratch --population 1000 --generations 100
tradebot evolve AAPL,MSFT --from-scratch --permutations 19 --folds 5
```

## Presets from the command line

* `--best-result`: set every setting for the best out-of-sample result, ignoring runtime (much slower than the defaults):
  bigger search, shuffled-price runs, walk-forward folds, limits scaled to the data. Flags you give yourself still win.
* `--auto-settings`: a search picks the settings for this data first (successive halving over many settings, ranked by
  validation score minus the overfitting gap; the newest `--test` part is kept out of it). Much slower than `--best-result`.
* `--effort good|best` goes with either one: `good` is a somewhat reasonable run time, `best` ignores it.

## Using the result

`--save NAME` writes the program as a plugin file (`~/.config/tradebot/plugins/NAME.py`, or `--plugin-dir`;
`--force` overwrites). It's ordinary strategy code with the rules in its docstring, so it works in `backtest`, the chart,
the Backtest lab and the bot, and you can read and edit it. Every number in the rules becomes a parameter (`n1`, `n2`, ...
for whole numbers, `c1`, ... for the rest). `tradebot optimize SYMBOL NAME` can then fine-tune them, with walk-forward if
you like. Exits that look at the open trade are evaluated trade by trade on the CPU; everything else is batched like any
strategy.
