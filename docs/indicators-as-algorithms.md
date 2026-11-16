# Indicators as algorithms

An **indicator** draws lines on a chart. It doesn't trade. This guide turns one into an **algorithm** (the code's name
for a strategy) that trades, and so shows up in the Backtest lab, the optimizer, the `vote` ensemble, the `pool`, the
combo builder, the chart's signal box and the live bot. The finished example is
[`examples/plugins/zscore_reversion.py`](../examples/plugins/zscore_reversion.py), covered by
[`tests/test_example_plugins.py`](../tests/test_example_plugins.py).

## How they differ

| | Indicator (`BaseIndicator`) | Algorithm (`BaseAlgorithm`) |
|---|---|---|
| Returns | named **outputs**: `{"value": ..., "upper": ...}`, one number per candle | one **target position** per candle |
| Method | `compute(cls, o, **params) -> dict` (a classmethod) | `signals(self, o, p)` |
| Used for | chart lines, panes, alerts, the Scanner and Screener | backtest, optimize, bot, vote, pool, combo |
| Parameters | `params = {"period": IntParam(...)}` | `params = {...}`, declared the same way |

The target position is `1` (fully long), `0` (flat), `-1` (short, only with `allow_short = True`) or a fraction. The
position decided at candle *t*'s close earns candle *t+1*'s return, in backtests and live alike.

An indicator output is a *number*; an algorithm needs *events*: "enter now", "leave now". The bridge is
`ops.latch(enter, exit)`: it switches to 1 on `enter`, back to 0 on `exit`, holds in between, and lets `exit` win when
both fire on the same candle. For a "long while the condition holds" rule use `ops.where(cond, 1.0, 0.0)` instead.

## Step by step

**1. Start from the indicator.** Here a z-score of the close: how many standard deviations it is from its moving average.

```python
from tradebot.core import ops
from tradebot.core.registry import IntParam
from tradebot.indicators.base import BaseIndicator, Style

class ZScore(BaseIndicator):
    name = "zscore"
    description = "Distance of the close from its moving average, in standard deviations"
    params = {"period": IntParam(20, 5, 200, "Candles in the average and the standard deviation.")}
    outputs = ("value", "oversold")
    overlay = False                      # its own pane under the price
    levels = (-2.0, 0.0, 2.0)            # reference lines
    styles = {"value": Style(color="accent", width=2)}
    alerts = {"oversold": "The z-score is below -2"}   # 0/1 per bar, hidden from the chart

    @classmethod
    def compute(cls, o, period):
        mean = ops.sma(o.close, period)
        std = ops.rolling_std(o.close, period)
        z = (o.close - mean) / std
        return {"value": z, "oversold": ops.flag(z < -2.0, z)}
```

**2. Wrap it.** Subclass `BaseAlgorithm`, declare `name`, `description` and `params`, and call
`Indicator.compute(o, ...)` inside `signals(o, p)`:

```python
from tradebot.algorithms.base import BaseAlgorithm
from tradebot.core.registry import Param

class ZScoreReversion(BaseAlgorithm):
    name = "zscore_reversion"
    description = "Buy when the close is `entry` standard deviations below its average, sell when it is back near it"
    params = {
        "period": IntParam(20, 5, 200, "Candles in the average the z-score is measured against."),
        "entry": Param(2.0, 0.5, 4.0, "Buy when the z-score falls below minus this."),
        "exit": Param(0.0, -1.0, 1.0, "Sell when the z-score is back above this."),
    }

    def signals(self, o, p):
        z = ZScore.compute(o, period=p["period"])["value"]
        return ops.latch(z < -p["entry"], z > p["exit"])

    def chart_indicators(self):
        return [ZScore(period=self.values["period"])]
```

Defining the class registers it (`__init_subclass__`), so there is nothing else to call.

**3. Try it.** Put the file where tradebot finds it (see [Where to put the file](#where-to-put-the-file)) and run:

```sh
tradebot list algorithms                       # zscore_reversion is in the list
tradebot backtest BTC/USDT zscore_reversion --bars 1500 -p entry=2.2
tradebot optimize BTC/USDT zscore_reversion --walk-forward 5
tradebot chart BTC/USDT --algo zscore_reversion
```

## Parameters

`Param(default, low, high, description)` is a float; `IntParam(default, low, high, description)` is a whole number
(window lengths, counts). The range is what the [optimizer](optimizers.md) searches, so make it sensible: wide enough to
find something, narrow enough not to waste the search. The description is the tooltip in the app; write it for a
beginner.

Indicator parameters are **passed through by hand**: the algorithm gets `p["period"]` and hands it to
`ZScore.compute(o, period=p["period"])`. You can share a definition when the names are the same
(`params = Ichimoku.params`, as the built-in `ichimoku_cloud` does), or declare your own and map them. The same names
work with `-p name=value` on the CLI, `--fix name=value` in `optimize`, and in the lab's form. Unknown names are
rejected with a list of the valid ones.

Floats can arrive as arrays (below). Whole numbers are always plain `int`, so they are safe as window lengths.

## What the chart shows

`chart_indicators()` returns **instances** of the indicators the chart should draw when the algorithm is the chart's
signal: `[ZScore(period=self.values["period"])]`. Buy and sell markers come from the position changes. The indicator
itself decides how it looks: `overlay` (price pane or its own pane), `levels` (reference lines), `styles` (colour, width
and dash per output; `"accent"`, `"up"`, `"down"`, `"muted"`, `"text"` and `"series[N]"` follow the theme), `labels`,
`fills`, `hidden`. Return `[]` (the default) to draw nothing extra. The same indicator can be added by hand from the
chart's **Indicators** menu, with the user's own colours on top.

## Staying vectorized

The optimizer evaluates a whole **population** of parameter sets in one call, on the GPU or a NumPy pool. For that,
`signals` must be written so that the same code works on arrays of shape `[candles]` (a normal backtest) and on arrays of
shape `[population, candles]`, where each float parameter is an array `[population, 1]` that broadcasts against the
candles (the optimizer, torch tensors included).

The rules:

* Use **only** `tradebot.core.ops` (`sma`, `ema`, `wilder`, `shift`, `rolling_max`, `rolling_min`, `rolling_std`,
  `window_sum`, `where`, `maximum`, `minimum`, `cross_above`, `cross_below`, `latch`, `flag`, ...) and plain arithmetic
  and comparisons. No `np.` or `math.` calls on the arrays, no `if` on array values (use `ops.where(cond, a, b)`), no
  `float(p["x"])` on a float parameter.
* **No Python loops over candles.**
* Calling other indicators' `compute` is fine: they follow the same rules.
* Don't change the arrays in place.

An algorithm that follows them stays at `vectorized = True` (the default) and runs everywhere. The repository's test
checks it by feeding the example a `[3, candles]` batch of parameters.

**When you can't** (event-driven logic, a state machine, a pattern detector that needs a loop), set
`vectorized = False`. Then `signals(o, p)` receives plain NumPy arrays and scalar parameters, and loops are fine. The
optimizer then evaluates the candidates one by one on the CPU (with `--backend auto` it selects the NumPy backend, as
the GPU would only add transfers): slower, but it works. `elliott_wave` takes about 3.5 ms per candidate for 1,500
candles. A `vote` or `pool` that contains such a member becomes loop-based as a whole.

Two optional flags:

* `period_batched = True`: your `signals` also accepts integer window lengths as `[population, 1]` arrays, which means it
  uses only ops that take an array period (`sma`, `ema`, `window_sum`, `shift`). The optimizer then batches candidates
  with different windows in one call. Only `macd_cross` and `sma_cross` do today; leave it off unless you test it.
* `poolable = False` keeps it out of the optimizer's automatic member selection (for composites, or when the parameters
  have no meaningful range).

## Avoiding look-ahead

The engine already delays: a position decided at the close of candle *t* earns candle *t+1*. So using candle *t*'s own
values in `signals` is correct. Look-ahead comes from using candles *after* *t*:

* **Negative shifts** (`ops.shift(x, -1)`) and anything centred on the current candle.
* **Full-history statistics**: `x / x.max()`, or normalising with the whole sample's mean or standard deviation.
* **Hindsight outputs**: pivots, zigzags and chart patterns that are only labelled once later candles confirm them. The
  Elliott wave labels are exactly that. That is why `elliott` has two kinds of output: the zigzag and wave labels are a
  **hindsight** view for the chart, while the `state` output only uses swings already confirmed at each candle, and is
  the one to trade on.
* **Levels that include the current candle** when you mean to trade a break of them: the Donchian `upper` of candle *t*
  contains candle *t*'s high, so `close > upper` can never be true. Compare with the previous candle's level:
  `o.close > ops.shift(ch["upper"], 1)`, as `examples/plugins/donchian_breakout.py` does.

A quick test for your own indicator or algorithm: the output on the first N candles must not change when more candles are
added. The example's test does this for the positions:

```python
assert np.array_equal(algo.run(df.iloc[:300]).to_numpy(), algo.run(df).to_numpy()[:300])
```

The evolver and the combo builder run the same kind of check and leave out indicator outputs that fail it.

## Using the result elsewhere

A registered algorithm is a strategy like any other:

* **Backtest lab, chart, CLI**: pick it by name.
* **`vote`**: `tradebot backtest AAPL vote --members zscore_reversion,sma_cross,rsi_threshold` (or tick it in the lab's
  *Members* card). Its parameters appear as `zscore_reversion.entry` and so on, so the whole ensemble can be optimized.
* **`pool`**: because it is poolable, `tradebot optimize BTC/USDT` considers it as a member the optimizer switches on and
  off. (A bare *indicator* can join the pool as a filter with a default rule, but only built-in indicators have one; your
  algorithm is the way to bring a plugin indicator into the pool.)
* **Combos**: the algorithm's name is a building block (`zscore_reversion & close > sma(200)`), and the plain indicator
  works without any algorithm as a condition: `zscore(20) < -2` (only outputs without look-ahead are allowed there). See
  [Strategies](strategies.md#the-combo-builder).
* **Scanner and Screener**: the *indicator* appears in the Scanner's picker; its `alerts` become filter chips.
* **Live**: `tradebot bot run BTC/USDT zscore_reversion --paper`.

## Where to put the file

tradebot imports every `*.py` file (not starting with `_`) from:

1. `~/.config/tradebot/plugins/`, or the folders in `$TRADEBOT_PLUGIN_DIR` (separated like `PATH`: `:` on Linux and
   macOS, `;` on Windows; it replaces the default folder);
2. installed packages that publish an entry point in the `tradebot.plugins` group (the standalone binary can't see
   those).

```sh
mkdir -p ~/.config/tradebot/plugins
cp examples/plugins/zscore_reversion.py ~/.config/tradebot/plugins/
tradebot list indicators      # zscore is there
```

A file that fails to import is logged and skipped; it can't take the app down. The Scanner's script editor
([Scanner](scanner.md#writing-your-own-indicator)) edits and reloads *indicator* files live. When two classes use the same
`name`, the later one wins and a warning is logged, so don't reuse built-in names (the `roc` indicator in
`examples/plugins/donchian_breakout.py` replaces the built-in one).

For an entry point, in your package's `pyproject.toml`:

```toml
[project.entry-points."tradebot.plugins"]
my_plugin = "my_package.plugin_module"
```

## Testing it

The repository's example is tested by loading the file through the real plugin loader into a temporary folder: it
registers, backtests, accepts a batch of parameters, has no look-ahead and works inside `vote` and the pool. Copy that
pattern for your own plugins. By hand, `tradebot backtest SYMBOL your_algorithm` is the quickest check.

Related: [Extending](extending.md) for the full list of indicator attributes, [Strategies](strategies.md) and
[How the optimizers work](optimizers.md).
