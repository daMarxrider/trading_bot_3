# Strategies

A **strategy** (the code calls it an *algorithm*) turns candles into a target position per candle: `1` long, `0` flat,
`-1` short (only for strategies that allow it; `--direction` trades any of them short, see
[Shorts and leverage](shorts-and-leverage.md)). `tradebot list algorithms` prints the strategies of your installation,
with their parameters and ranges. Plugins add more ([Extending](extending.md)); strategies saved by
[`evolve`](evolve.md) or the combo builder appear here too.

## Built-in strategies

| Name | Does | Parameters |
|---|---|---|
| `rsi_threshold` | Buys when the RSI falls below `lower`, sells when it rises above `upper` | `period`, `lower` (33), `upper` (66) |
| `macd_cross` | Long while the MACD line is above its signal line | `fast`, `slow`, `signal` |
| `ichimoku_cloud` | Long while the close is above both cloud spans | the Ichimoku periods |
| `sma_cross` | Long while the fast SMA is above the slow SMA | `fast`, `slow` |
| `bollinger_reversion` | Buys a close below the lower band, exits at the middle band | `period`, `width` |
| `range` | Buys at or below `buy_price`, sells at or above `sell_price` (or below `stop_price`) | absolute prices |
| `elliott_wave` | Trades Elliott wave 3 (and 5) breakouts, see below | wave and stop settings |
| `halving_cycle` | In the market from `buy_day` to `sell_day` days after a Bitcoin halving, see below | `buy_day`, `sell_day` |
| `vote` | Majority vote over several strategies | `threshold` plus each member's parameters |
| `pool` | Like `vote`, but the optimizer picks the members | see [optimizers](optimizers.md#picking-the-strategies-the-pool) |
| `combo` | Entry and exit rules from strategies and indicator conditions, see below | `--entry`, `--exit` |

Each parameter has a tooltip in the app explaining it in plain words (they come from the `Param` descriptions).

## vote

`vote` enters when at least `threshold` (default 0.7) of its members are long, exits when at least `threshold` are flat
and holds otherwise. Member parameters are exposed as `<member>.<param>` (for example `rsi_threshold.lower`), so the
ensemble can be optimized as a whole. The default members are `rsi_threshold`, `macd_cross` and `ichimoku_cloud`; pick
others with `--members` (CLI) or the **Members** card (lab):

```sh
tradebot backtest AAPL vote --members rsi_threshold,sma_cross,bollinger_reversion
```

## The combo builder

A **combo** is a strategy built from other strategies and indicator conditions, without code. It has an entry rule and
an exit rule:

```sh
tradebot backtest AAPL combo --entry "elliott_wave & close > sma(200)" --exit "macd_cross | stop_atr(2)"
tradebot backtest AAPL combo --entry "weighted(rsi_threshold:0.5, macd_cross:0.3, sma_cross:0.2) >= 0.6"
tradebot combo save trend_dip --entry "close > sma(200) & rsi(14) < 35" --exit "rsi(14) > 55 | stop_pct(5)"
tradebot combo list
```

Building blocks:

* a **strategy** name stands for its signal: in the entry rule it holds while that strategy is in the market, in the exit
  rule while it is out (its exit). `!name` inverts. Parameters as arguments: `rsi_threshold(lower=30)`.
* an **indicator value**, compared: `close > sma(200)`, `rsi(14) < 30`, `elliott.state == 3`,
  `sma(50) > sma(200) * 1.02`. `name(args).output` picks an output. Only outputs that never change when later candles
  arrive are allowed (no hindsight). Prices: `open high low close typical`. Arithmetic: `+ - * /`.
* an indicator **rule**: the indicator name alone (`rsi`, `sma(period=100)`) is its pool rule; an alert output alone
  (`sma.cross_up`) is true on its bars.
* logic: `&` / `and`, `|` / `or`, `!` / `not`, brackets; `vote(a, b, c) >= 0.6` and `weighted(a:0.5, b:0.3) >= 0.6`.
* **risk exits** (exit rule, joined with `|`): `stop_pct(5)`, `stop_atr(2)`, `take_profit(10)`, `take_profit_atr(3)`,
  `trail_pct(5)`, `trail_atr(3)`, `max_hold(20)`. Take-profits can scale out: `take_profit(5, 0.5) | take_profit(10)` sells
  half at +5 % and the rest at +10 %.
* a **limit entry** (entry rule, joined with `&`): `limit_pct(1)` / `limit_atr(0.5)` buys with a limit order below the
  close of the candle on which the rest holds, for the next candle.
* a **higher timeframe**: `tf=` on an indicator, rule or strategy runs it on longer candles, only closed ones:
  `close > sma(200, tf=1d)` on hourly candles, `rsi(14, tf=4h) < 30`, `macd_cross(tf=1d)`. Its parameters are
  `sma_1d.period` and so on. See [multi-timeframe indicators](multi-timeframe.md).
* `entry` in the exit rule is the entry rule: `!entry | stop_pct(5)`. Without an exit rule a combo exits when the entry
  rule stops holding.

Semantics: everything is evaluated on closed candles; entry and exit are **latched** (once in, only the exit rule or a
risk exit gets out); an entry and an exit on the same candle: the exit wins; a condition whose indicator is still
warming up is neither true nor false.

**Fills** (`--fills close|intrabar`, the lab's *Intrabar* box): at `close` (default) stops, take-profits and limit entries
are compared with the close and filled there. `intrabar` fills them inside the candle against its high and low, at the
level, or at the open when the candle opens beyond it (a gap); a stop and a take-profit in one candle: the stop counts.

Every number in a rule is a tunable parameter, namespaced like the vote's (`rsi_threshold.lower`, `sma.period`,
`stop_atr.mult`, weights `w.<member>`, `threshold`), so the optimizer tunes members, weights and thresholds together;
comparison constants (`< 30`) stay fixed. **Saved combos** live in `combos.toml` next to `config.toml` and are registered
as strategies named after them: they work in `list`, `backtest`, `optimize`, `bot`, `chart`, the GUI and inside other
combos. The lab's *Combo builder* card fills the two rules by picking blocks, a logic and a threshold, and saves them.

## Elliott waves

**Indicator `elliott`** finds swings with an ATR-based zigzag (`swing_atr` sets the minimum swing size) and checks every
swing sequence against the hard Elliott rules: wave 2 never retraces more than 100 % of wave 1; wave 3 is never the
shortest of waves 1, 3 and 5; wave 4 doesn't overlap wave 1; in a correction, C extends beyond A. Each match gets a
Fibonacci conformity score from 0 to 1. Charts draw the zigzag with labels 0 to 5 and A to C. These labels are a
**hindsight** view (they change as new candles arrive), so the indicator also provides a **causal** `state` output for
trading and screening:

* `3`, `4`, `5`: that wave is in progress; `10`: the impulse is complete; `11` to `13`: waves A to C of the correction;
* negative values: the same for a bearish structure.

`state` only uses swings that were already confirmed at each bar. Use it in rules (`elliott.state == 3`) and as a
screener column.

**Strategy `elliott_wave`** (long-only): after a valid wave 1-2 whose wave-2 retracement lies within
`wave2_min`-`wave2_max`, it buys the breakout above the wave-1 high; stop below the wave-2 low minus `stop_atr` x ATR,
target `target_ext` x wave 1, and it also exits when the wave-3 top is confirmed. `trade_wave5` adds the same logic
after a valid wave 1-4. It needs a loop over the candles, so the optimizer evaluates it on the CPU.

## Bitcoin halving cycle

Every 210,000 blocks the Bitcoin block reward halves. At the target of 600 s per block that is 1458.33 days, the
"4-year rule". Blocks come faster while the hashrate grows, so every halving so far came early, by 1.3 % to 9.6 %.
`tradebot halving` shows the past halvings and how each estimator would have predicted them from the earlier ones; the
best (the average time per block of all completed epochs) forecasts the next halving, and `--live` adds a forecast from
the current chain tip's pace (asks the Blockstream API).

**Indicator `halving`** is a calendar clock, so it works on any timestamped candles:

* `days`: days since the latest halving;
* `days_to_next`: days until the next halving *as predicted at that bar from the earlier halvings only*, so a backtest
  never sees the future;
* `phase` (not drawn): `days` as a fraction of the predicted cycle.

**Strategy `halving_cycle`** is long from `buy_day` to `sell_day` days after a halving; with `sell_day` below `buy_day`
the trade spans the halving. The defaults (buy at day 850, sell at day 530) come from the past cycles: tops 371 to 548
days after a halving, bottoms 777 to 924. With only four cycles so far, optimized days fit the past easily: use
`--walk-forward` and compare with buy & hold.

## Indicators

`tradebot list indicators` lists them. An indicator draws lines on the chart and can carry **alerts** (named 0/1 states
the [Scanner](scanner.md) filters on); it does not trade. To make one trade, wrap it in a strategy:
[Indicators as algorithms](indicators-as-algorithms.md).
