# Brokers and live trading

> Live mode places real orders with real money. tradebot is provided as is, without warranty, and nothing here is
> financial advice ([Disclaimer and licence](legal.md)). Run a strategy as a dry run, and on a testnet (`--sandbox`),
> before you use `--live`.

## One execution model

A strategy returns a target position per candle. The backtester, the optimizer and the live bot all use the same model:
the position decided at candle *t*'s close earns candle *t+1*'s return. On each **closed** candle the bot evaluates the
strategy and buys or sells to match its target. A target above 0.5 buys (up to `--stake`, the maximum quote-currency
value per entry); once in, a lower target sells that share of the holding (a combo's stepped take-profits scale out) and
a target of 0 sells everything.

One difference from the backtest: the bot acts at the close. A combo's intrabar stop, which the backtest fills inside a
candle, is sold by the bot at that candle's close.

## Brokers

Every broker has the same `get_ticker`, `get_klines`, `place_order`, `cancel_order` and `get_balance` methods plus an
execution mode:

* `live`: orders go to the venue's API. The crypto venues use ccxt: **Binance, KuCoin, Pionex**.
* `semi_auto`: there is no trading API (**Scalable Capital**, **Trade Republic**). Orders become order intents for you to
  execute by hand, and holdings are read from a CSV (`symbol,quantity,avg_cost`).
* `read_only`: tracking only.
* `paper`: a simulated account, see below.

`tradebot list brokers` shows what your installation has. Credentials go under `[brokers.<name>]` in `config.toml` (or
[Settings](settings.md), stored in plain text), or in the environment as `TRADEBOT_<BROKER>_<KEY>`, for example
`TRADEBOT_BINANCE_SECRET`; the environment wins. Use API keys **without withdrawal rights**.

```toml
[brokers.binance]
api_key = "..."
secret = "..."

[brokers.kucoin]
api_key = "..."
secret = "..."
password = "..."            # KuCoin API passphrase

[brokers.scalable]
holdings_csv = "~/finance/scalable.csv"   # columns: symbol,quantity,avg_cost
```

Trade Republic login needs the optional `pytr` package; Scalable Capital's official CLI `sc` is a separate tool (see
[Getting started](getting-started.md#install)). Adding a broker: [Extending](extending.md#brokers).

## Running a bot

```sh
tradebot bot run ETH/USDT sma_cross --broker binance --tf 1h --stake 50                  # dry run: nothing is sent
tradebot bot run ETH/USDT sma_cross --broker binance --tf 1h --stake 50 --sandbox        # exchange testnet
tradebot bot run ETH/USDT sma_cross --broker binance --tf 1h --stake 50 --live          # real orders
tradebot bot run ETH/USDT sma_cross --tf 1h --stake 50 --paper                           # simulated account
tradebot bot run ETH/USDT sma_cross --once                                               # evaluate one candle and exit
```

The first `--live` run shows the risk notice and asks you to accept it; your answer is saved as
`live_risk_accepted = "<date>"` in `config.toml`, so it asks only once. Parameters, `--members`, `--entry`/`--exit` and
`--fills` work as in `backtest`. A single-symbol `bot run` doesn't remember an open position across a restart: check the broker's positions
before starting again.

Between two closes the bot watches the price over a websocket (polling when there is none; `--no-stream` turns it
off). With a combo that fills intrabar (`--fills intrabar`) it sells as soon as the candle reaches a stop or a
take-profit, as the backtest does, instead of at the close; risk limits are checked on every price too, so
`--close-on-halt` sells right away. Decisions to enter still happen at the close. See
[Streaming prices](streaming.md).

## Paper trading

`--paper` (or `--broker paper`) trades a simulated account: orders fill at the last price of the real data source minus
the proportional fee (0.1 % by default). Cash, positions and the order history are saved after every order to
`paper_account.json` in the app's config folder, written atomically, so a restart resumes with the same holdings.

```sh
tradebot paper show                 # equity, P&L, positions and the latest orders (--orders N)
tradebot paper reset --balance 5000 # start over (-y: don't ask)
```

```toml
[brokers.paper]
starting_balance = 10000
currency = "USDT"
fee = 0.001
```

There is no GUI view of the paper account yet.

## Risk guards and the kill switch

All limits are optional and off at 0. Money is in the quote currency; drawdown is measured in percent of the stake on
realised plus open profit, from its peak since the start.

| Option | Effect |
|---|---|
| `--max-daily-loss` | Halt for the rest of the UTC day after losing this much |
| `--max-drawdown` | Halt when equity falls this many % of the stake below its peak (until `bot resume` or a restart) |
| `--max-exposure` | Reject entries that would push the position above this value |
| `--max-order` | Reject orders above this value |
| `--price-band` | Reject entries when the exchange price is this many % off the last close |
| `--cooldown-losses`, `--cooldown-minutes` | Pause entries (60 minutes by default) after this many losing trades in a row |
| `--close-on-halt` | Sell the open position when a limit halts the bot |

**Kill switch**: `tradebot bot kill` (or the **Kill switch** button in the app's status bar) stops every running `bot run` at its
next check; `tradebot bot resume` turns it off again, but a bot that already halted needs a restart. It is a file named
`KILL` in the config folder, so the CLI, the GUI and bots in other processes share it. Every trigger is logged and sent
to the `risk` notification channel.

## Notifications

Bot trades (`trade`), failed bot steps (`error`), risk-guard stops (`risk`) and scanner hits (`scanner`, with
`tradebot scan --notify`) can be pushed to Telegram, a Discord webhook, a generic JSON webhook, email or the desktop
(`notify-send` on Linux, `osascript` on macOS, else the app's tray). Configure them under
[Settings](settings.md#notifications) or by hand, one table per channel; `events` filters the kinds (default: all), and
`enabled = false` mutes a channel:

```toml
[notify.phone]
type = "telegram"
token = "$TELEGRAM_TOKEN"   # "$NAME" or "env:NAME" reads the environment, keeping the secret out of the file
chat_id = "12345678"
events = ["trade", "error"]

[notify.mail]
type = "email"              # also: discord (url), webhook (url), desktop
host = "smtp.example.com"
port = 587                  # 465: SSL, otherwise STARTTLS
user = "me@example.com"
password = "$SMTP_PASSWORD"
to = "me@example.com"
```

Sends run in the background with a 10 s timeout; a failure is only logged (never with the secrets) and can't stop a bot.
`tradebot notify test` or the test button sends to every enabled channel. Code raises an event with
`tradebot.notify.notify("risk", title, text)` and adds a channel type with `@notify.channel(...)`.

## Journal

Every order a bot places or simulates is stored in the [journal](journal.md); `--no-journal` switches that off.

## Pairs

`tradebot bot pair-run LEG_A LEG_B` trades the gap of two assets, long one and short the other: see
[Pairs trading](pairs-trading.md#live), including how it remembers an open pair across restarts.

## Position sizing

`bot run --sizing MODEL` buys `stake` times the size the sizing model gives: see
[Market regimes and position sizing](regimes-and-sizing.md#position-sizing).
