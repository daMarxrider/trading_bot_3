# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Chart loadouts: named sets of indicators and a signal that can be saved and applied again.

A loadout is plain JSON, kept in ``settings.json`` (:class:`tradebot.config.Settings`)::

    {"indicators": [{"name": "bbands", "params": {"period": 20, "width": 2.0}}, ...],
     "signal": {"name": "vote", "params": {...}, "members": ["rsi_threshold", "macd_cross"]} | null,
     "line": false}

Restoring is forgiving: indicators or algorithms that no longer exist (e.g. a removed plugin)
are skipped and reported, and unknown or out-of-range parameters fall back to valid values.
"""

from __future__ import annotations

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.indicators import INDICATORS

KEY = "chart.loadouts"


def capture(indicators, algorithm, line: bool) -> dict:
    """The chart's current setup as a loadout."""
    signal = None
    if algorithm is not None:
        members = getattr(algorithm, "members", None)  # vote ensembles
        signal = {"name": algorithm.name, "params": dict(algorithm.values),
                  "members": [m.name for m in members] if members else None}
    return {"indicators": [{"name": ind.name, "params": dict(ind.values)} for ind in indicators],
            "signal": signal, "line": bool(line)}


def restore(data: dict):
    """``(indicators, algorithm or None, line, skipped names)`` for a saved loadout."""
    indicators, skipped = [], []
    for spec in data.get("indicators", []):
        name = spec.get("name")
        if name not in INDICATORS:
            skipped.append(name)
            continue
        cls = INDICATORS.get(name)
        indicators.append(cls(**_valid(cls.params, spec.get("params", {}))))
    algorithm = None
    signal = data.get("signal")
    if signal:
        members = signal.get("members") or None
        missing = [n for n in [signal.get("name"), *(members or [])] if n not in ALGORITHMS]
        if missing:
            skipped += missing
        else:
            algorithm = build_algorithm(signal["name"], members=members)
            algorithm = algorithm.with_params(**_valid(algorithm.param_space(), signal.get("params", {})))
    return indicators, algorithm, bool(data.get("line", False)), skipped


def _valid(space: dict, given: dict) -> dict:
    """Known parameters only, clipped into their ranges."""
    return {k: space[k].clip(v) for k, v in given.items() if k in space and isinstance(v, (int, float))}


class LoadoutStore:
    """Saved loadouts by name, in insertion order; in memory when there's no ``Settings`` (e.g. tests)."""

    def __init__(self, settings=None):
        self.settings = settings
        self._memory: dict = {}

    def all(self) -> dict:
        return dict(self.settings.get(KEY, {}) if self.settings is not None else self._memory)

    def save(self, name: str, loadout: dict) -> None:
        saved = self.all()
        saved[name] = loadout  # overwriting keeps the position
        self._put(saved)

    def delete(self, name: str) -> None:
        saved = self.all()
        saved.pop(name, None)
        self._put(saved)

    def _put(self, saved: dict) -> None:
        if self.settings is not None:
            self.settings.set(KEY, saved)
        else:
            self._memory = saved
