# Reference

## File locations

| What | Linux | macOS | Windows |
|---|---|---|---|
| `config.toml` | `~/.config/tradebot/` | `~/.config/tradebot/` | `%USERPROFILE%\.config\tradebot\` |
| `combos.toml` (saved combos) | next to `config.toml` | same | same |
| `plugins/`, `themes/` | `~/.config/tradebot/plugins/`, `.../themes/` | same | same |
| `settings.json`, `paper_account.json`, `journal.sqlite`, `alerts.json`, `pair_bots/`, `KILL` (kill switch) | `$XDG_CONFIG_HOME/tradebot/` (usually `~/.config/tradebot/`) | `~/Library/Application Support/tradebot/` | `%APPDATA%\tradebot\` |
| Cache (candles, symbol lists) | `$XDG_CACHE_HOME/tradebot/` (usually `~/.cache/tradebot/`) | `~/Library/Caches/tradebot/` | `%LOCALAPPDATA%\tradebot\Cache\` |
| Shell completion lookups | in the cache folder | same | same |

`config.toml` is the file you edit; `settings.json` is what the app remembers by itself. The whole cache folder is safe to
delete. Saved evolver results (`--save`) go to the first plugin folder.

## Environment variables

| Variable | Effect |
|---|---|
| `TRADEBOT_CONFIG` | path of `config.toml` (its folder also holds `combos.toml`) |
| `TRADEBOT_PLUGIN_DIR` | plugin folder(s), separated like `PATH`; replaces `~/.config/tradebot/plugins` |
| `TRADEBOT_PAIR_STATE_DIR` | folder of the live pair bot's state files (default: `pair_bots/` in the config folder) |
| `TRADEBOT_JOURNAL` | path of the trade journal database (default `journal.sqlite` in the config folder) |
| `TRADEBOT_CACHE_DIR` | cache folder, overriding the OS default |
| `TRADEBOT_<BROKER>_<KEY>` | a broker credential, e.g. `TRADEBOT_BINANCE_SECRET`, `TRADEBOT_KUCOIN_PASSWORD`; wins over `config.toml` |
| `ALPHAVANTAGE_API_KEY`, `FMP_API_KEY` | keys of the optional data providers (without a key they are skipped) |
| `XDG_CONFIG_HOME`, `XDG_CACHE_HOME` | Linux base folders for `settings.json` and the cache |
| `APPDATA`, `LOCALAPPDATA` | Windows base folders for `settings.json` and the cache |
| `CUDA_VISIBLE_DEVICES` | the usual CUDA variable, honoured by the GPU optimizer |
| `$NAME` / `env:NAME` as a value in `[notify.*]` | read that environment variable instead of writing the secret in the file |

## `config.toml`

Edited by hand or by [Settings](settings.md) (which only changes its own lines and keeps your comments).

```toml
theme = "graphite"              # graphite, paper, glass, dark, light, or the file name (without .json) of your own theme
touch_mode = "auto"             # auto (on when a touchscreen is present), on, off
gpu_rendering = false           # chart through OpenGL, needs a restart
live_risk_accepted = "2026-11-01"   # written when you accept the live-trading notice

[watchlists]                    # edited in the chart sidebar, or here by hand
Main = ["AAPL", "MSFT", "SAP.DE", "BTC/USDT", "ETH/USDT"]
"Long term" = ["SAP.DE", "ETH/USDT"]   # quote names with spaces

[brokers.binance]
api_key = "..."
secret = "..."

[brokers.kucoin]
api_key = "..."
secret = "..."
password = "..."                # KuCoin API passphrase

[brokers.scalable]
holdings_csv = "~/finance/scalable.csv"   # columns: symbol,quantity,avg_cost

[brokers.paper]
starting_balance = 10000
currency = "USDT"
fee = 0.001

[tools]
sc = "/opt/scalable/sc"         # optional; default: found on PATH

[backtest]
slippage_bps = 0                # per side, in basis points; a number, or { crypto = 5, stock = 2 }

[cache]
enabled = true                  # false: keep candles in memory only
max_mb = 500

[notify.phone]                  # one table per channel, see Brokers and live trading
type = "telegram"
token = "$TELEGRAM_TOKEN"
chat_id = "12345678"
events = ["trade", "error"]
```

| Key | Meaning |
|---|---|
| `theme` | the theme at startup; the Settings dialog writes it |
| `touch_mode` | finger-sized controls: `auto`, `on`, `off` |
| `gpu_rendering` | draw the chart through OpenGL (experimental) |
| `live_risk_accepted` | the date you accepted the live-trading notice; delete it to see the notice again |
| `[watchlists]` | named lists of symbols. An older top-level `watchlist = [...]` is read as the list "Main" and moved here the first time you change a watchlist |
| `[brokers.<name>]` | credentials per broker; the available keys depend on the broker (`api_key`, `secret`, `password`, `holdings_csv`, ...) |
| `[brokers.paper]` | `starting_balance`, `currency`, `fee` of the paper account |
| `[tools]` | paths of external tools by key (`sc`) |
| `[backtest]` `slippage_bps` | default slippage per side, a number or a table per asset class (`crypto`, `stock`, `fx`) |
| `[cache]` | `enabled`, `max_mb` |
| `[notify.<name>]` | `type` (`telegram`, `discord`, `webhook`, `email`, `desktop`), the channel's options, `events` (`trade`, `error`, `risk`, `scanner`), `enabled` |

Suggestions, the screener's default symbols, the News page, the CLI's `screen` and `news` defaults and the cache warm-up
at startup use the symbols of all watchlists. Telegram options: `token`, `chat_id`; Discord and webhook: `url`; email:
`host`, `port`, `user`, `password`, `to`, `from`.

## `settings.json`

UI state the app saves by itself on every change. Delete the file to reset it. Keys include:

| Key | Holds |
|---|---|
| `chart.session` | the symbol, timeframe, indicators and signal of the last session |
| `chart.loadouts` | saved loadouts |
| `chart.drawings` | drawings per symbol |
| `chart.styles` | your indicator line colours |
| `chart.autofit`, `chart.log`, `chart.volume_overlay` | the chart toggles |
| `chart.sidebar.open`, `chart.sidebar.category`, `chart.watchlist` | the sidebar state and the shown watchlist |
| `backtest.autofit`, `backtest.log`, `backtest.panes`, `backtest.columns` | Backtest lab toggles and pane sizes |
| `evolve.autofit`, `evolve.history_autofit`, `evolve.log`, `evolve.panes`, `evolve.plots`, `evolve.columns` | the same for Evolve |
| `scanner.*` | the Scanner's pane sizes and choices (source, indicator, inputs, filters) |

## Command-line summary

Commands: `list`, `chart`, `backtest`, `montecarlo`, `optimize`, `evolve`, `positions`, `portfolio`, `screen`, `scan`, `news`,
`pair`, `pair-backtest`, `pair-scan`, `halving`, `device`, `gui`, and the groups `bot` (`run`, `kill`, `resume`,
`pair-run`), `cache` (`info`, `clear`), `journal` (`show`, `note`, `export`), `alert` (`price`, `indicator`, `scan`, `list`, `rm`), `watch`, `bot pair-state`, `combo` (`save`, `list`, `delete`), `notify` (`test`), `paper` (`show`, `reset`).
All options: [CLI reference](cli-reference.md).

## Built-in names

* Strategies: `rsi_threshold`, `macd_cross`, `ichimoku_cloud`, `sma_cross`, `bollinger_reversion`, `range`,
  `halving_cycle`, `elliott_wave`, `vote`, `pool`, `combo` ([Strategies](strategies.md)).
* Brokers: `binance`, `kucoin`, `pionex`, `scalable`, `traderepublic`, `paper`.
* Data sources: Binance, KuCoin, Kraken, Yahoo Finance, Alpha Vantage, Financial Modeling Prep ([Data](data.md)).
* Index lists for the Scanner and `tradebot scan --index`: `sp500`, `nasdaq100`, `dow30`, `dax40`, `ftse100`,
  `crypto_top100`.
