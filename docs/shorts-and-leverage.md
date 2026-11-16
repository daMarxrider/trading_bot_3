# Shorts and leverage

Out of the box every strategy is long or flat. You can also backtest and optimize a strategy **short**, or
**long and short**, with **leverage**. The costs of holding those positions are in the backtest too: borrow
fees for stock shorts, funding for crypto perpetual futures, and liquidation.

```sh
tradebot backtest AAPL bollinger_reversion --direction long_short --borrow-fee 0.02
tradebot backtest BTC/USDT macd_cross --direction long_short --leverage 3 --funding history
tradebot backtest BTC/USDT sma_cross --leverage 5 --maintenance-margin 0.01 --funding 0.0001
tradebot optimize BTC/USDT rsi_threshold --direction short --leverage 2
```

In the app, the Backtest lab has **Direction**, **Leverage**, **Borrow fee** and **Funding** fields under
Position sizing. They apply to the backtest and the optimizer. The live bot never opens shorts or leveraged
positions.

## Direction

* `long`: the default, unchanged.
* `short`: short on the strategy's sell signal, flat otherwise.
* `long_short`: long on the buy signal, short on the sell signal.

The short side mirrors the long one. For crossovers and the RSI levels, the mirror is the time the long
version is out of the market, counted from its first entry, so `long_short` is stop and reverse. Where the
mirror is different, the strategy has its own short rule:

* `bollinger_reversion` shorts a close above the upper band and covers at the middle band.
* `ichimoku_cloud` is short below the cloud and flat inside it.

Combos work the same way, as long as they fill at the close. Intrabar fills (`--fills intrabar`) can't be
combined with shorts yet. Sizing (`--sizing`) sits on top: `--sizing fixed -p size.fraction=0.5
--direction long_short` holds +0.5 or -0.5.

## How a leveraged or short position is backtested

A long position without leverage works as before. A short position, leverage or funding switches the
backtest to a margin model:

* **Margin.** A target position `p` (between -1 and 1) puts `|p|` of the equity up as margin for a position
  worth `leverage × |p|` of the equity. This is isolated margin: the rest of the account isn't at risk.
* **Fixed units.** While the target stays the same, the number of units is fixed at the entry price. A
  losing position therefore gets more leveraged, as it does on an exchange. The plain long backtest instead
  rebalances to the same fraction every candle.
* **Liquidation.** A position is liquidated in the first candle whose worst price (the low for a long, the
  high for a short) brings its margin down to the maintenance margin (`--maintenance-margin`, 0.5% of the
  notional by default). The whole margin is lost, and the position stays flat until the strategy's target
  changes. A gap through the liquidation price loses the same amount, no more.
  * 10x long: liquidated about 9.5% below the entry.
  * 3x long: about 33% below.
  * A short without leverage: when the price about doubles.
  * A long without leverage can't be liquidated.
* **Fees** and slippage are charged on the traded notional, so a 3x position pays three times the fee.
* **Borrow fee** (`--borrow-fee`, a yearly fraction): charged every candle on a short's notional.
* **Funding** (`--funding`): every position pays its notional times the rate. A positive rate means longs
  pay and shorts receive. Give either a constant rate per 8 hours (`0.0001` = 0.01%, about 11% a year) or
  `history`. With `history`, the settled rates of the symbol's perpetual (`BTC/USDT` → `BTC/USDT:USDT` on
  Binance USD-M) are downloaded and summed per candle. The optimizer only takes a constant rate.

The result has a **liquidations** metric (how often a position was liquidated), and the command line lists
the dates.

### Simplifications

* The fee on a change of the target ignores how far the notional drifted since the entry.
* Borrow fees and funding come out of the equity, not out of the margin, so they don't move the liquidation
  price.
* Margin tiers aren't modelled: one maintenance rate applies to every position size.
* The buy & hold and monthly-investing baselines stay long and unleveraged.

The portfolio backtest (`--portfolio`) stays long-only without leverage.
