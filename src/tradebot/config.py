# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""User configuration (``~/.config/tradebot/config.toml`` or ``$TRADEBOT_CONFIG``).

Example::

    theme = "glass"

    [watchlists]
    Main = ["AAPL", "MSFT", "BTC/USDT", "ETH/USDT"]
    "Long term" = ["SAP.DE", "ETH/USDT"]

    [brokers.binance]
    api_key = "..."
    secret = "..."

    [brokers.scalable]
    holdings_csv = "~/finance/scalable_holdings.csv"

Secrets can also come from the environment: ``TRADEBOT_BINANCE_API_KEY``,
``TRADEBOT_KUCOIN_PASSWORD``, ... (``TRADEBOT_<BROKER>_<KEY>``), which take precedence.

UI state the app remembers itself (e.g. toggles) lives in :class:`Settings`, a JSON file in
the OS config folder. The exceptions are ``theme`` and ``[watchlists]``, which the app writes here with
:func:`save_config_value`, changing only those lines and keeping everything else.

An older config's top-level ``watchlist = [...]`` is read as the list "Main", and moved to ``[watchlists]``
the first time a watchlist is changed in the app.
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

WATCHLISTS, MAIN_LIST = "watchlists", "Main"  #: the config table of the named watchlists, the first list's name
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "SAP.DE", "BTC/USDT", "ETH/USDT", "SOL/USDT"]


def config_path() -> Path:
    return Path(os.environ.get("TRADEBOT_CONFIG", "~/.config/tradebot/config.toml")).expanduser()


def load_config() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def save_config_value(key: str, value: str | bool | int | float | list[str] | None, table: str | None = None,
                      rename: str | None = None) -> bool:
    """Set ``key = value`` at the top level or in ``[table]`` of the config file, preserving everything else.

    Strings, booleans, numbers and lists of strings are written as TOML values (JSON spells them the same way);
    a key that isn't a bare TOML key (e.g. one with spaces) is quoted.

    Only that key's entry changes (a trailing comment on it is kept, a multi-line array is replaced whole); a
    missing key is added at the end of its section (top-level keys before the first ``[table]``), a missing table
    at the end of the file, a missing file is created. ``rename``: an existing key whose entry this one replaces,
    in its place. ``value=None`` removes the key. Returns False (and logs) if it can't be written.
    """
    path = config_path()
    try:
        text = path.read_text() if path.is_file() else ""
    except OSError as exc:
        log.warning("could not read %s: %s", path, exc)
        return False
    # a JSON string is a valid TOML basic string
    entry = None if value is None else f"{_toml_key(key)} = {json.dumps(value, ensure_ascii=False)}"
    before, section, after = _section(text, table)
    if section is None:  # no such table yet
        if entry is None:
            return True
        new = text + ("\n" if text and not text.endswith("\n") else "") + ("\n" if text else "") + f"[{table}]\n{entry}\n"
    else:
        new = before + _set_line(section, key, entry, gap=bool(after) if table is None else None, rename=rename) + after
    if new == text:
        return True
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


def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key, ensure_ascii=False)


def _set_line(section: str, key: str, entry: str | None, gap: bool | None, rename: str | None = None) -> str:
    """``section`` with ``key``'s entry (or ``rename``'s) replaced by ``entry`` (None: removed), or ``entry`` appended.

    ``gap``: top level only, whether a blank line must separate the new key from the first table (None: a table).
    """
    span = (_entry_span(section, rename) if rename else None) or _entry_span(section, key)
    if span is not None:
        start, end, comment = span
        return section[:start] + ("" if entry is None else entry + comment + "\n") + section[end:]
    if entry is None:
        return section
    if gap is None:  # a table: right after its last key, keeping the blank lines before the next table
        body = section.rstrip("\n")
        return (body + "\n" if body else "") + entry + "\n" + section[len(body) + 1:]
    return (section if section.endswith("\n") or not section else section + "\n") + entry + "\n" + ("\n" if gap else "")


def _entry_span(section: str, key: str) -> tuple[int, int, str] | None:
    """``(start, end, trailing comment)`` of ``key``'s entry: from its line's start to after its line end, a
    multi-line array included. The key may be written bare or quoted."""
    forms = [json.dumps(key, ensure_ascii=False), f"'{key}'"] + ([key] if _toml_key(key) == key else [])
    m = re.search(rf"(?m)^[ \t]*(?:{'|'.join(map(re.escape, forms))})[ \t]*=[ \t]*", section)
    if m is None:
        return None
    end = _value_end(section, m.end())
    while end > m.end() and section[end - 1] in " \t":
        end -= 1
    rest = re.compile(r"(?P<comment>[ \t]*#[^\n]*)?[ \t]*(\n|$)").match(section, end)
    if rest is None:  # something this doesn't understand: leave it to the TOML check
        return None
    return m.start(), rest.end(), rest.group("comment") or ""


def _value_end(text: str, i: int) -> int:
    """Where the TOML value starting at ``text[i]`` ends: brackets are matched across lines, strings and comments
    inside them skipped."""
    depth = 0
    while i < len(text):
        ch = text[i]
        if ch in "\"'":
            quote = ch * 3 if text.startswith(ch * 3, i) else ch
            j = i + len(quote)
            while j < len(text) and not text.startswith(quote, j) and (len(quote) == 3 or text[j] != "\n"):
                j += 2 if ch == '"' and text[j] == "\\" else 1
            i = j + len(quote)
            if depth == 0:
                return min(i, len(text))
            continue
        if ch == "#" or ch == "\n":
            if depth == 0:
                return i
            if ch == "#":
                i = text.find("\n", i)
                i = len(text) if i < 0 else i
                continue
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _section(text: str, table: str | None) -> tuple[str, str | None, str]:
    """``(before, section, after)``: the top-level keys (``table=None``) or the body of ``[table]``;
    section is None if the table doesn't exist."""
    if table is None:
        top, rest = _split_top_level(text)
        return "", top, rest
    header = re.search(rf"(?m)^[ \t]*\[[ \t]*{re.escape(table)}[ \t]*\][ \t]*(#[^\n]*)?(\n|$)", text)
    if header is None:
        return text, None, ""
    nxt = re.search(r"(?m)^[ \t]*\[", text[header.end():])
    stop = header.end() + nxt.start() if nxt else len(text)
    return text[: header.end()], text[header.end(): stop], text[stop:]


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


def watchlists() -> dict[str, list[str]]:
    """The named watchlists (#67): ``[watchlists]``, or until that exists the top-level ``watchlist`` (or the
    default) as "Main"."""
    config = load_config()
    table = config.get(WATCHLISTS)
    lists = {str(name): [str(s) for s in symbols] for name, symbols in table.items() if isinstance(symbols, list)} \
        if isinstance(table, dict) else {}
    return lists or {MAIN_LIST: [str(s) for s in config.get("watchlist", DEFAULT_WATCHLIST)]}


def watchlist(name: str | None = None) -> list[str]:
    """The symbols of one watchlist, or of all of them (de-duplicated, in order)."""
    lists = watchlists()
    if name is not None:
        if name not in lists:
            raise KeyError(f"no watchlist {name!r}; there are: {', '.join(lists)}")
        return list(lists[name])
    return list(dict.fromkeys(s for symbols in lists.values() for s in symbols))


def save_watchlist(name: str, symbols: list[str], rename: str | None = None) -> bool:
    """Write one watchlist to ``[watchlists]``; ``rename``: the list's old name (it keeps its place)."""
    return _migrate_watchlist() and save_config_value(name, symbols, WATCHLISTS, rename=rename)


def delete_watchlist(name: str) -> bool:
    return _migrate_watchlist() and save_config_value(name, None, WATCHLISTS)


def _migrate_watchlist() -> bool:
    """Before the first change: the top-level ``watchlist`` (or the default) becomes ``[watchlists]`` "Main"."""
    try:
        config = load_config()
    except (OSError, tomllib.TOMLDecodeError) as exc:
        log.warning("not saving watchlists: can't read %s (%s)", config_path(), exc)
        return False
    if isinstance(config.get(WATCHLISTS), dict):
        return True
    return save_config_value(MAIN_LIST, watchlists()[MAIN_LIST], WATCHLISTS) and save_config_value("watchlist", None)


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
    if os.environ.get("TRADEBOT_CACHE_DIR"):
        return Path(os.environ["TRADEBOT_CACHE_DIR"]).expanduser()
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
