# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""External tools tradebot can call, which users install themselves (#58, #59).

A configured path lives in config.toml's ``[tools]`` table, keyed by the tool's ``key``::

    [tools]
    sc = "/opt/scalable/sc"

Without one, the tool is looked up on ``PATH`` by its ``command``. Adding a tool to :data:`TOOLS` is all
the settings dialog needs to offer it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from tradebot.config import load_config, save_config_value

TABLE = "tools"


@dataclass(frozen=True)
class Tool:
    key: str  #: config key in [tools]
    name: str
    command: str  #: looked up on PATH when no path is configured
    url: str  #: official download page
    used_for: str


TOOLS = [
    Tool("sc", "Scalable Capital CLI", "sc", "https://github.com/ScalableCapital/scalable-cli/releases",
         "Scalable Capital broker"),
]


def tool(key: str) -> Tool:
    return next(t for t in TOOLS if t.key == key)


def configured_path(key: str) -> str:
    """The path set in config.toml, or "" for "look it up on PATH"."""
    return str(load_config().get(TABLE, {}).get(key, "") or "")


def set_path(key: str, path: str) -> bool:
    """Save a tool's path; an empty one removes it (back to PATH lookup). False if it couldn't be written."""
    path = path.strip()
    return save_config_value(key, os.path.expanduser(path) if path else None, table=TABLE)


def locate(key: str, path: str | None = None) -> str | None:
    """The executable to run: ``path`` (default: the configured one) if it's an executable file, otherwise the
    tool's command on PATH when no path is set. None if not found."""
    path = configured_path(key) if path is None else path.strip()
    if path:
        path = os.path.expanduser(path)
        return path if os.path.isfile(path) and os.access(path, os.X_OK) else None
    return shutil.which(tool(key).command)
