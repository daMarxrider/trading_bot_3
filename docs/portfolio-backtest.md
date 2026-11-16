# Portfolio backtest

A normal backtest, and the multi-symbol optimizer, look at each symbol on its own. A portfolio backtest
runs **one account** that trades the same strategy on several symbols at once. The symbols share the
capital, so a trade in one means less money for the others.

```sh
tradebot backtest AAPL,MSFT,NVDA,KO,JNJ sma_cross --portfolio
tradebot backtest AAPL,MSFT,NVDA,KO,JNJ sma_cross --max-positions 3 --allocation volatility --rebalance month
tradebot backtest BTC/USDT,ETH/USDT,SOL/USDT combo --entry "close>sma(200)" --max-positions 2 -w crypto
```

`--max-positions` or `--watchlist` turn on `--portfolio` by themselves. `--trades-csv` and `--report` work
too. The trade list gets a `symbol` column. In the app, the **Basket** page does the same thing.

## How it trades

The strategy runs on each symbol's own candles, with the full history for its warm-up. After each candle
the account decides what to hold:

* **Slots.** `--max-positions N` allows at most N positions at a time; the default is one per symbol.
  Every slot is worth 1/N of the account, and an empty slot stays in cash. A position you already hold keeps
  its slot. Free slots go to new signals, the strongest first, and on a tie, the symbol listed first.
* **Allocation.** This sets how the held slots split their capital:
  * `equal`: the same share each.
  * `signal`: in proportion to the signal's strength. This only differs from `equal` when the
    strategy gives fractional positions, for example with `--sizing` or a vote.
  * `volatility`: inverse volatility over the last 20 candles. Calm symbols get more, so each position adds
    about the same risk.

  With `equal` and `volatility`, a fractional signal also scales its own position: a strategy that is half
  in gets half its share.
* **Rebalancing.** Between rebalance dates a position drifts with its price. A new entry gets its target
  weight, as far as the cash allows, and an exit is sold. On each rebalance date (`candle`, `week`, `month`
  or `quarter`) every held position is set back to its target weight. `none`, the default, never
  rebalances. A position is also resized whenever the strategy's own target for it changes.

Orders fill at the close of the candle that produced them, as in the single-symbol backtest. The fee and
the slippage are charged on every weight change. There is no leverage: what the account holds never adds up
to more than the account.

The symbols are lined up on all of their candle times, but only inside the span that every symbol covers.
Between a symbol's own candles its last close carries over, so a stock stays flat over a crypto weekend.
Intrabar fills (`--fills intrabar`) are not modelled here: every fill is at the close.

## What you get

* The portfolio's metrics next to **buy & hold of the same basket**: equal weights bought on the first
  candle and never rebalanced. `vs buy & hold` is the XIRR difference, and a red line warns when the
  portfolio does not beat the basket.
* `avg positions`: how many positions were held on average. `exposure`: the share of candles with
  anything invested.
* Per symbol: the money it made or lost, costs included (these add up to the account's profit), the
  share of the time it was held, its average weight while held, and its number of trades.
* The equity curve of both, in the terminal (`--no-plot` hides it) or on the Basket page.
