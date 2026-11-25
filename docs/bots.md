# Bots: several at once

> Live mode places real orders with real money. tradebot is provided as is, without warranty, and nothing here is
> financial advice ([Disclaimer and licence](legal.md)). Run every bot as a dry run, then on the paper account, before
> you give it `mode = "live"`.

`tradebot bot run` runs one strategy on one symbol. A **bot set** runs several of them at once, with limits they share,
and is saved in a file so the same set starts again after a reboot. Each bot trades exactly like `bot run`
([One execution model](live-trading.md#one-execution-model)).

## The bot set: `bots.toml`

The set lives in `bots.toml` next to `config.toml` (`~/.config/tradebot/bots.toml` on Linux; `TRADEBOT_BOTS` points
elsewhere). `tradebot bots add`, `tradebot bots limits` and the app's **Bots** page write it; you can also edit it by
hand (the app rewrites it whole, so comments in it are not kept).

```toml
[limits]                      # shared by all bots of one process; missing or 0: off
max_total_exposure = 1500     # all open positions together, after an entry (quote currency)
max_open_positions = 3        # bots holding a position at the same time
max_daily_loss = 80           # realised + open loss of all bots in one UTC day: every bot halts until tomorrow
close_on_halt = false         # also sell the open positions when that daily limit halts them

[[bots]]
name = "eth-trend"
symbol = "ETH/USDT"
algorithm = "sma_cross"
timeframe = "1h"
stake = 200                   # max quote-currency value of one entry
params = { fast = 10, slow = 30 }
risk = { max_drawdown = 10 }  # this bot's own limits: the options of `bot run`

[[bots]]
name = "btc-swing"
symbol = "BTC/USDT"
algorithm = "macd_cross"
timeframe = "4h"
stake = 500
mode = "paper"
```

| Key | Default | Meaning |
|---|---|---|
| `name`, `symbol`, `algorithm` | required | A unique name, what to trade and the strategy (`tradebot list`) |
| `broker` | `binance` | Where the candles come from and, in live mode, where the orders go |
| `timeframe`, `stake` | `1h`, `100` | As `--tf` and `--stake` of `bot run` |
| `mode` | `dry` | `dry` (nothing is sent), `paper` (the [paper account](live-trading.md#paper-trading)) or `live` |
| `params`, `members`, `entry`, `exit`, `fills`, `sizing` | | As `-p`, `--members`, `--entry`, `--exit`, `--fills`, `--sizing` |
| `sandbox` | `false` | The exchange testnet (ccxt brokers) |
| `enabled` | `true` | Started by `bots run` and the app's **Start all** |
| `risk` | | `max_daily_loss`, `max_drawdown`, `max_exposure`, `max_order_value`, `price_band`, `cooldown_losses`, `cooldown_minutes`, `close_on_halt` ([Risk guards](live-trading.md#risk-guards-and-the-kill-switch)) |

Two paper or live bots may not trade the same asset on the same account at once (they would sell each other's coins):
starting the second one fails with a message. Dry-run bots don't share anything and may overlap.

## Shared limits

The per-bot limits still apply; the `[limits]` table adds limits for the whole set:

* **max_total_exposure** rejects an entry when the value of all bots' open positions plus the new one would exceed it.
* **max_open_positions** rejects an entry while that many bots already hold a position.
* **max_daily_loss** halts every bot for the rest of the UTC day once all of them together have lost that much since
  midnight, realised plus open; with `close_on_halt` each bot sells its position at its next check.
* The [kill switch](live-trading.md#risk-guards-and-the-kill-switch) stops every bot, as before.

A rejected entry shows up in the bot's events as `rejected: …`, a halt as `halted: …`, and both are sent to the `risk`
[notification](live-trading.md#notifications) channel.

## From the terminal

```sh
tradebot bots add eth-trend ETH/USDT sma_cross --tf 1h --stake 200 -p fast=10 -p slow=30 --max-drawdown 10
tradebot bots add btc-swing BTC/USDT macd_cross --tf 4h --stake 500 --mode paper
tradebot bots limits --max-total-exposure 1500 --max-open-positions 3 --max-daily-loss 80
tradebot bots list                 # state, position, P&L and the last event of each bot
tradebot bots run                  # every enabled bot, until Ctrl+C
tradebot bots run eth-trend --once # one bot, one candle
tradebot bots run --live           # also the bots with mode = "live" (asks for the risk notice once)
tradebot bots stop eth-trend       # or --all: the process that runs it stops it within a second
tradebot bots disable btc-swing    # leave it out of `bots run`; `enable` takes it back
tradebot bots remove btc-swing
```

`bots run` evaluates every bot right away and then just after each of its candles closes; one scheduler steps all bots
in turn. Bots with `mode = "live"` are skipped unless you pass `--live`.

### Start after a reboot

`bots run` is what you start at boot. On Linux, a systemd user unit:

```ini
# ~/.config/systemd/user/tradebot-bots.service
[Unit]
Description=tradebot bots
After=network-online.target

[Service]
ExecStart=%h/.local/bin/tradebot bots run
Restart=on-failure
RestartSec=60

[Install]
WantedBy=default.target
```

```sh
systemctl --user enable --now tradebot-bots
loginctl enable-linger "$USER"   # keep it running while you are logged out
```

Add `--live` to `ExecStart` only for live bots you have tested. On Windows use the Task Scheduler ("At startup"), on
macOS a launchd agent, or cron's `@reboot` anywhere it exists.

A bot resumes where it stopped: after every step it saves its position, entry price, P&L and the last 30 events to
`bots/<name>.json` in the config folder, and loads them at the next start (as long as its symbol and mode are the
same). Check the broker's positions after a crash all the same.

## In the app: the Bots page

**Account › Bots** is the dashboard of the set.

* The row at the top: bots running, open positions, their value (exposure), the P&L of the running bots (realised plus
  open, with ▲/▼) and the shared limits, or the reason they halted.
* The table: one row per bot with its mode, state (running, stopped, halted, error, or *other process* when
  `tradebot bots run` runs it), position, entry and last price, P&L and the last event. Hover a row for the details,
  double-click it to start or stop it.
* **Start**, **Stop**, **Start all**, **Stop all** act on the selected bot or on the whole set. **Stop** also works for
  a bot another process runs.
* **Events** lists the last events of the selected bot (or of all bots): buy, sell, hold, rejected entries and halts.
* **Bot** edits the selected bot or, after **New bot**, a new one; **Save bot** writes `bots.toml`. Editing a running bot
  stops it, start it again afterwards. Combo rules, members and sizing are kept as they are in the file.
* **Shared limits** sets the `[limits]` table, and **Start the enabled bots when the app opens** starts the set with the
  app.

The app runs **dry-run and paper bots only**: neither sends a real order, and the status bar chip says so
(`DRY-RUN BOTS · NO REAL ORDERS`, `PAPER BOTS · NO REAL ORDERS`). Live bots run in a terminal with
`tradebot bots run --live`; the page shows them, with their state and P&L, and can stop them. Closing the app stops the
bots it runs; they resume when you start them again. Every order of a paper bot goes to the [journal](journal.md).
