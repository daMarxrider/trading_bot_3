# Settings

**⚙ Settings** (bottom of the left navigation) opens a dialog with the categories on the left. Changes are saved to
`config.toml` right away; only the lines you change are rewritten, your comments and other settings stay, a missing
file is created, and an empty field removes its line.

## Appearance

* **Theme**: **Glass** (translucent frosted panels over a soft gradient backdrop), **Modern Dark** and **Light**,
  plus your own. It applies right away (`theme = "glass"` in the config).
* **GPU rendering (experimental)**: draws the chart through OpenGL for smoother panning (`gpu_rendering = true`). Off
  by default, applies after a restart. Without OpenGL (no driver, a VM, a remote desktop) the chart uses the normal
  renderer and the status bar says so.

Your own theme is a JSON file in `~/.config/tradebot/themes/`. Any token you leave out falls back to the `base` theme
(`dark` or `light`):

```json
{"name": "Midnight", "base": "dark", "accent": "#ff6ad5", "bg": "#0b0b1a"}
```

Tokens: `bg`, `panel`, `panel_border`, `text`, `muted`, `accent`, `up`, `down`, `grid`, `input_bg`, `selection`,
`radius`, `glass`, `font` and `series` (the list of indicator colours).

## Integrations

* **Third-party tools**: the path to each external tool tradebot can use, such as the Scalable Capital CLI (`sc`,
  saved under `[tools]`). Leave it empty to find the tool on `PATH`. Each tool shows whether it was found, and a
  download link if it wasn't.
* **Broker API keys**: the credentials of every broker (API key, secret, passphrase, holdings CSV, ...), saved under
  `[brokers.<name>]`. They're stored in **plain text**. A key set as an environment variable
  (`TRADEBOT_<BROKER>_<KEY>`) takes precedence and is shown greyed out. Use API keys without withdrawal rights.

## Notifications

One box per channel type (Telegram bot, Discord webhook, generic webhook, email (SMTP), desktop notification) with an
**Enabled** switch, the channel's fields and the **events** to send: `trade`, `error`, `risk`, `scanner`. **Send test
message** sends to every enabled channel and shows which worked. To keep a secret out of the file, write `$ENV_NAME`
as the value and set that environment variable. Details and the by-hand format: [Brokers and live
trading](live-trading.md#notifications).

## Data & cache

Switch the disk cache on or off and set its maximum size (50 to 50,000 MB). The page shows how much chart data and
how many symbol lists are cached, and where (**Open folder**). *Clear chart data*, *Clear symbol lists* and *Clear
all* delete them after asking. See [Data providers, timeframes and the cache](data.md#the-cache).

## Other buttons of the navigation

* **⛔ Kill switch**: stops every running live bot at its next check, the same as `tradebot bot kill`. Untick it to
  resume (a bot that already halted needs a restart).
* **ⓘ About**: version, the disclaimers and the licence, the same text as [Disclaimer and licence](legal.md).
