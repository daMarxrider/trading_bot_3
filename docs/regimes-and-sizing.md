# Market regimes and position sizing

## Market regimes

The `regime` indicator classifies every candle as a bull, bear or sideways trend (the close up or down more than
`threshold` percent over `period` candles) and as high or low volatility (ATR above or below its average over five times
`vol_period`). It only uses earlier candles. Parameters: `period` (50), `threshold` (5), `vol_period` (14).

* **Chart**: add `regime`; the background is shaded green (bull), red (bear) and amber (high volatility).
* **Combo filter**: `--entry "rsi_threshold & regime.bull"`, `regime.low_vol`, `regime(100, 10).sideways`, or the numeric
  trend, `regime > 0`. Outputs: `bull`, `bear`, `sideways`, `high_vol`, `low_vol`, `trend` (-1, 0, 1), `volatility` (0, 1).
  See [Strategies](strategies.md#the-combo-builder).
* **Evolve** offers `regime.trend` and `regime.volatility` as primitives like any other indicator output.
* **Breakdown**: `tradebot backtest SYMBOL combo ... --regimes` prints return, Sharpe, exposure and trades per regime; the
  HTML report and the Backtest lab's report card show the same table.

## Position sizing

A strategy says *when* to be in the market; `--sizing` decides *how much* to put on, as a layer over any strategy
(`backtest`, `optimize`, `chart`, `bot run`; the [Backtest lab](backtest-lab.md) has a "Position sizing" box).

| Model | Size |
|---|---|
| `fixed` | a fraction of the capital (`size.fraction`) |
| `vol` | volatility target: the position's annual volatility aims at `size.target_vol` (realised volatility over `size.window` candles), so quiet markets get a bigger position |
| `risk` | fixed risk per trade: the stop sits `size.stop_atr` ATRs below the entry and the size makes a stop-out cost `size.risk` of the capital; fixed at the entry |
| `kelly` | `size.kelly_fraction` of the Kelly fraction of the strategy's recent returns (the fraction itself until 10 candles in the market have been seen) |

The settings are ordinary parameters named `size.*`: set them with `-p size.target_vol=0.15`, fix them with `--fix`, or let
the [optimizer](optimizers.md) tune them. Sizing never adds leverage (the size is at most 1; leverage is a separate setting, see
[Shorts and leverage](shorts-and-leverage.md)), so sizing only lowers exposure;
compare `exposure` and `max_drawdown` with and without. The live bot follows the strategy's signal and buys `stake` times
the size.

```sh
tradebot backtest BTC/USDT sma_cross --sizing vol -p size.target_vol=0.2
```
