# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""User configuration (``~/.config/tradebot/config.toml`` or ``$TRADEBOT_CONFIG``).

Example::

    watchlist = ["AAPL", "MSFT", "BTC/USDT", "ETH/USDT"]
    theme = "glass"

    [brokers.binance]
    api_key = "..."
    secret = "..."

    [brokers.scalable]
    holdings_csv = "~/finance/scalable_holdings.csv"

Secrets can also come from the environment: ``TRADEBOT_BINANCE_API_KEY``,
``TRADEBOT_KUCOIN_PASSWORD``, ... (``TRADEBOT_<BROKER>_<KEY>``), which take precedence.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "SAP.DE", "BTC/USDT", "ETH/USDT", "SOL/USDT"]


def config_path() -> Path:
    return Path(os.environ.get("TRADEBOT_CONFIG", "~/.config/tradebot/config.toml")).expanduser()


def load_config() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def broker_credentials(name: str) -> dict:
    creds = dict(load_config().get("brokers", {}).get(name, {}))
    prefix = f"TRADEBOT_{name.upper()}_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            creds[key[len(prefix):].lower()] = value
    return creds


def watchlist() -> list[str]:
    return list(load_config().get("watchlist", DEFAULT_WATCHLIST))
