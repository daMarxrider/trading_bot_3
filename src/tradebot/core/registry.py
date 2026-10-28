# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Name -> class registries and the dynamic plugin loader.

Indicators, algorithms and brokers register themselves automatically when
their class is defined (see ``__init_subclass__`` in the base classes), so a
plugin is just a ``.py`` file that subclasses a base class. Plugins are
discovered from:

* ``$TRADEBOT_PLUGIN_DIR`` (``os.pathsep``-separated) or ``~/.config/tradebot/plugins``
* installed packages exposing the ``tradebot.plugins`` entry-point group
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import sys
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Generic, Iterator, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class Param:
    """Tunable parameter: default value plus the search range used by the optimizer."""

    default: float
    low: float
    high: float
    integer: bool = False
    description: str = ""

    def clip(self, value: float) -> float:
        value = min(max(value, self.low), self.high)
        return int(round(value)) if self.integer else float(value)


def IntParam(default: int, low: int, high: int, description: str = "") -> Param:  # noqa: N802
    return Param(default, low, high, integer=True, description=description)


def resolve_params(space: dict[str, Param], given: dict) -> dict:
    """Fill defaults and reject unknown names; values are cast to int where declared."""
    unknown = set(given) - set(space)
    if unknown:
        raise ValueError(f"unknown parameter(s) {sorted(unknown)}; expected {sorted(space)}")
    out = {}
    for key, spec in space.items():
        value = given.get(key, spec.default)
        out[key] = int(value) if spec.integer else float(value)
    return out


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, cls: type[T]) -> type[T]:
        name = getattr(cls, "name", None)
        if not name or inspect.isabstract(cls):
            return cls
        existing = self._items.get(name)
        if existing is not None and existing is not cls:
            log.warning("%s %r re-registered by %s.%s", self.kind, name, cls.__module__, cls.__qualname__)
        self._items[name] = cls
        return cls

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {', '.join(self.names())}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[type[T]]:
        return iter(self._items[n] for n in self.names())

    def __len__(self) -> int:
        return len(self._items)


def default_plugin_dirs() -> list[Path]:
    env = os.environ.get("TRADEBOT_PLUGIN_DIR")
    if env:
        return [Path(p).expanduser() for p in env.split(os.pathsep) if p]
    return [Path("~/.config/tradebot/plugins").expanduser()]


def load_plugins(dirs: list[Path] | None = None) -> list[str]:
    """Import every plugin file and entry point; returns the loaded module names.

    A broken plugin is logged and skipped so it cannot take the platform down.
    """
    loaded = []
    for directory in dirs if dirs is not None else default_plugin_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            mod_name = f"tradebot_plugin_{path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(mod_name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = module
                spec.loader.exec_module(module)
                loaded.append(mod_name)
            except Exception:
                sys.modules.pop(mod_name, None)
                log.exception("failed to load plugin %s", path)
    if dirs is None:
        for ep in entry_points(group="tradebot.plugins"):
            try:
                ep.load()
                loaded.append(ep.name)
            except Exception:
                log.exception("failed to load plugin entry point %s", ep.name)
    return loaded
