# Disclaimer and licence

This page repeats the text of the **About** dialog of the desktop app, of `tradebot --version` and of the notice before
the first live order (`src/tradebot/legal.py`). The documentation check fails when it drifts apart.

## Not affiliated

tradebot is independent software. It is not affiliated with, endorsed, sponsored or approved by any broker, exchange or data provider it supports (such as Binance, KuCoin, Pionex, Scalable Capital, Trade Republic or Yahoo). Their names and trademarks belong to their owners and are used only to say which services tradebot works with.

## No warranty, no liability

tradebot is provided "as is", without any warranty, under the GNU GPL v3 (sections 15 and 16). It is not financial advice, and backtest or optimizer results don't predict future returns. You use it at your own risk: to the extent permitted by law, the author is not liable for any loss, including losses from bugs, wrong signals, failed, delayed, duplicated or wrongly sized orders, broker or API outages, wrong market data, or misconfiguration. You are responsible for your accounts and API keys and for checking every order.

## Live trading

Live mode places real orders with real money. Try the strategy as a dry run (and on a testnet with `--sandbox`) first.

The first `tradebot bot run ... --live` shows the notice and asks you to accept it. Your answer is saved as
`live_risk_accepted = "<date>"` in `config.toml`, so it asks only once. See [Brokers and live trading](live-trading.md).

## Data and third-party services

Market data comes from third-party services under their own terms; Yahoo Finance data, for example, is meant for personal
use. Check each broker's API terms before automating trades with it. tradebot works with some external tools (`pytr`, the
Scalable Capital CLI `sc`) but never ships them; install them yourself, and their own licences and terms apply.

## Licence

tradebot is licensed under the **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`); the full text is in the
`LICENSE` file of the repository and of every release archive. Every release archive also contains
`THIRD_PARTY_NOTICES.txt` with the licences of all bundled libraries. Source code:
[github.com/daMarxrider/trading_bot_3](https://github.com/daMarxrider/trading_bot_3).
