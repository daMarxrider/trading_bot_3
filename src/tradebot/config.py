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

UI state the app remembers itself (e.g. toggles) lives in :class:`Settings`, a JSON file in
the OS config folder, so the hand-edited config file is never rewritten.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tomllib
from pathlib import Path

log = logging.getLogger(__name__)

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


def os_config_dir() -> Path:
    """The OS's per-user config folder for tradebot: %APPDATA%, ~/Library/Application Support or XDG."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or "~/AppData/Roaming"
    elif sys.platform == "darwin":
        base = "~/Library/Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "tradebot"


def os_cache_dir() -> Path:
    """The OS's per-user cache folder for tradebot (safe to delete; it's rebuilt on demand)."""
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or "~/AppData/Local").expanduser() / "tradebot" / "Cache"
    if sys.platform == "darwin":
        return Path("~/Library/Caches/tradebot").expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "tradebot"


class Settings:
    """Persistent UI state, saved as JSON on every change.

    A missing, unreadable or corrupt file just means defaults, and a failed save is
    logged, so settings can never stop the app.
    """

    def __init__(self, path: Path | None = None):
        self.path = path or os_config_dir() / "settings.json"
        try:
            self._data = json.loads(self.path.read_text())
            if not isinstance(self._data, dict):
                raise ValueError("not a JSON object")
        except FileNotFoundError:
            self._data = {}
        except (OSError, ValueError) as exc:
            log.warning("ignoring settings file %s: %s", self.path, exc)
            self._data = {}

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
            tmp.replace(self.path)  # atomic: a crash mid-write never leaves a half-written file
        except OSError as exc:
            log.warning("could not save settings to %s: %s", self.path, exc)
