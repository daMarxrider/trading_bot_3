# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Disclaimers shown by ``tradebot --version``, the GUI's About dialog and before the first live order (#58).

Accepting the live-trading risk once is remembered as ``live_risk_accepted = "<date>"`` in config.toml.
"""

from __future__ import annotations

from datetime import date

from tradebot.config import load_config, save_config_value

NOT_AFFILIATED = (
    "tradebot is independent software. It is not affiliated with, endorsed, sponsored or approved by "
    "any broker, exchange or data provider it supports (such as Binance, KuCoin, Pionex, Scalable Capital, "
    "Trade Republic or Yahoo). Their names and trademarks belong to their owners and are used only to say "
    "which services tradebot works with."
)

NO_WARRANTY = (
    'tradebot is provided "as is", without any warranty, under the GNU GPL v3 (sections 15 and 16). '
    "It is not financial advice, and backtest or optimizer results don't predict future returns. "
    "You use it at your own risk: to the extent permitted by law, the author is not liable for any loss, "
    "including losses from bugs, wrong signals, failed, delayed, duplicated or wrongly sized orders, "
    "broker or API outages, wrong market data, or misconfiguration. You are responsible for your accounts "
    "and API keys and for checking every order."
)

DISCLAIMER = NOT_AFFILIATED + "\n\n" + NO_WARRANTY

LIVE_RISK = (
    "Live mode places real orders with real money. " + NO_WARRANTY
    + " Try the strategy as a dry run (and on a testnet with --sandbox) first."
)


def live_risk_accepted() -> bool:
    return bool(load_config().get("live_risk_accepted"))


def accept_live_risk() -> bool:
    """Remember the acceptance in config.toml; False if the file couldn't be written."""
    return save_config_value("live_risk_accepted", date.today().isoformat())
