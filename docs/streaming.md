# Streaming prices

Candles come from REST requests: the chart fetches them on **Load**, the bot once per candle close. Streaming adds
the prices in between, for the chart's **Live** button and for the live bot (`tradebot bot run`).

## Where the prices come from

| Symbol | Stream | Fallback |
|---|---|---|
| Crypto (`BTC/USDT`) | The exchange's websocket through ccxt pro (bundled with ccxt): the candle stream where the exchange has one, else the ticker on top of one REST snapshot of the candle | Polling the exchange's candles |
| Stocks and ETFs (`AAPL`, `SAP.DE`) | Yahoo Finance's price stream (only during market hours) | Polling Yahoo's candles |

The bot streams from its broker's exchange (Binance for `--broker binance`, the testnet with `--sandbox`); the
chart and the paper broker from the first exchange of the data sources, Binance by default. Polling asks for the
newest candles every 5 seconds, straight from the source and past the cache.

## Reconnects and fallback

A dropped connection is reopened after 1 second, then 2, 4 and so on (at most 30 seconds). After three failed
attempts in a row the stream polls instead and tries the websocket again five minutes later. A connection that
delivered prices before it dropped starts the count over. The chart's Live button shows a warning icon while it
polls or reconnects; `bot run` prints every change (`price stream: polling (websocket failed: ...)`).

## The bot inside the candle

Without streaming, the bot acts at the candle close, so a combo's intrabar stop that the backtest fills inside a
candle is sold at that candle's close. With streaming (the default; `--no-stream` turns it off):

* **Stops and take-profits** of a combo with `--fills intrabar` are sold as soon as the forming candle's low reaches
  the stop or its high a take-profit. The levels are the ones the backtest fills against: worked out at each close
  from the candles so far (entry price, highest close for trailing stops, ATR). Take-profits that scale out sell
  their share; the rest of the position keeps its stop.
* **Risk limits** (daily loss, drawdown, the kill switch) are checked on every price with the open profit at that
  price; with `--close-on-halt` a halt sells at once.
* **Entries**, the exit rule and `max_hold` still act at the close, as in the backtest. A limit entry (`limit_pct`,
  `limit_atr`) is bought at the close of the candle it would have filled on.

The bot sells at market, so the fill is the price when the level was seen, which can be worse than the level itself
in a fast market (the backtest fills at the level, or at the open after a gap). Combos with `--fills close` behave as
before: their levels are checked on the close.

## In code

```python
from tradebot.data.stream import open_stream

stream = open_stream("BTC/USDT", "1m")            # or broker=..., engine=..., exchange_id="kraken"
stream.on_tick(lambda t: print(t.price, t.candle))  # runs on the stream's thread
stream.on_state(lambda state, note: print(state, note))
stream.start()
...
stream.stop()
```

`MarketStream` takes any client with ccxt pro's `watch_ohlcv` / `watch_ticker` interface and a poll function, so a
new source is a small adapter (see `YahooStreamClient` in `tradebot/data/stream.py`).
