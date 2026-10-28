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
the OS config folder. The one exception is ``theme``: choosing a theme in the app writes it
here with :func:`save_config_value`, which changes only that line and keeps everything else.
"""

from __future__ import annotations

import json
import logging
import os
import re
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


def save_config_value(key: str, value: str) -> bool:
    """Set a top-level string ``key = "value"`` in the config file, preserving everything else.

    Only that key's line changes (a trailing comment on it is kept); a missing key is added before
    the first ``[table]``; a missing file is created. Returns False (and logs) if it can't be written.
    """
    path = config_path()
    try:
        text = path.read_text() if path.is_file() else ""
    except OSError as exc:
        log.warning("could not read %s: %s", path, exc)
        return False
    entry = f"{key} = {json.dumps(value)}"  # a JSON string of plain text is a valid TOML basic string
    top, rest = _split_top_level(text)
    line = re.compile(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=[^\n#]*?(?P<comment>[ \t]*#[^\n]*)?$")
    if line.search(top):
        top = line.sub(lambda m: entry + (m.group("comment") or ""), top, count=1)
    else:
        top = (top if top.endswith("\n") or not top else top + "\n") + entry + "\n" + ("\n" if rest else "")
    new = top + rest
    try:
        tomllib.loads(new)  # never write a file the app couldn't read back
    except tomllib.TOMLDecodeError as exc:
        log.warning("not saving %s to %s: the result would be invalid TOML (%s)", key, path, exc)
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(new)
        tmp.replace(path)
    except OSError as exc:
        log.warning("could not save %s to %s: %s", key, path, exc)
        return False
    return True


def _split_top_level(text: str) -> tuple[str, str]:
    """``(top-level part, tables part)``: keys before the first ``[table]`` header are top-level."""
    m = re.search(r"(?m)^[ \t]*\[", text)
    return (text, "") if m is None else (text[: m.start()], text[m.start():])


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
