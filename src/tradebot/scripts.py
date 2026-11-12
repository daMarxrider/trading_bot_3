# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Scanner scripts (#115): indicator plugin files that can be edited and reloaded while the app runs.

A script is an ordinary indicator plugin (a ``BaseIndicator`` subclass in a ``.py`` file in the first plugin
folder), so it also loads at the next start, in the CLI and on the other pages. :func:`reload_script` is the
live-edit step: it compiles the source, imports it under its ``tradebot_plugin_<stem>`` module name, dry-runs
every indicator it defines and, on any failure, puts the registry back as it was.

Scripts are trusted local code, the same as any plugin: nothing is sandboxed.
"""

from __future__ import annotations

import inspect
import re
import sys
import traceback
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tradebot.core.registry import default_plugin_dirs, resolve_params
from tradebot.core.series import OHLC
from tradebot.indicators.base import INDICATORS, BaseIndicator, is_screenable

DRY_RUN_BARS = 500
MODULE_PREFIX = "tradebot_plugin_"

TEMPLATE = '''\
"""My indicator for the Scanner and the chart.

How indicators work: see the docstring at the top of tradebot/indicators/base.py
(params, outputs, levels, labels, hidden, styles, units and alerts).
Save with Ctrl+S: the script is checked and reloaded at once.
"""

from tradebot.core import ops
from tradebot.core.registry import IntParam
from tradebot.indicators.base import BaseIndicator


class {cls}(BaseIndicator):
    name = "{name}"  # what the picker lists and the CLI uses; a new name registers a new indicator
    description = "Close against its moving average"
    params = {{"period": IntParam(20, 2, 200)}}  # numeric inputs, shown in the settings dialog
    outputs = ("average", "above")  # the arrays compute() returns
    alerts = {{"above": "Close is above the average"}}  # outputs that are true/false conditions

    @classmethod
    def compute(cls, o, period):
        # o.open, o.high, o.low, o.close are arrays; only use values of this bar and earlier ones
        average = ops.sma(o.close, period)
        return {{"average": average, "above": ops.flag(o.close > average, average)}}  # NaN while warming up
'''


def plugin_dir() -> Path:
    """The folder scripts are saved in: the first plugin folder."""
    return default_plugin_dirs()[0]


def module_name(stem: str) -> str:
    return MODULE_PREFIX + stem


def script_path(stem: str) -> Path:
    return plugin_dir() / f"{stem}.py"


def is_script(cls: type[BaseIndicator]) -> bool:
    return cls.__module__.startswith(MODULE_PREFIX)


def source_path(cls: type[BaseIndicator]) -> Path | None:
    """The file a script indicator was loaded from."""
    file = getattr(sys.modules.get(cls.__module__), "__file__", None)
    return Path(file) if file else None


def free_stem(base: str) -> str:
    """``base``, or ``base_2``, … until neither a file nor an indicator has that name."""
    stem, n = base, 1
    while script_path(stem).exists() or stem in INDICATORS or module_name(stem) in sys.modules:
        n += 1
        stem = f"{base}_{n}"
    return stem


def new_script() -> tuple[Path, str]:
    """A path (not written yet) and the template source for a new script."""
    stem = free_stem("my_indicator")
    cls = "".join(w.capitalize() for w in stem.split("_"))
    return script_path(stem), TEMPLATE.format(cls=cls, name=stem)


def copy_of(cls: type[BaseIndicator]) -> tuple[Path, str]:
    """A path (not written yet) and the source of ``cls`` under a new name, for "Copy to my scripts"."""
    module = sys.modules[cls.__module__]
    source = inspect.getsource(cls)
    stem = free_stem(f"{cls.name}_copy")
    new_cls = "".join(w.capitalize() for w in stem.split("_"))
    source = re.sub(rf"^class {cls.__name__}\(", f"class {new_cls}(", source, count=1, flags=re.M)
    source = re.sub(r'^(\s+)name = "[^"]*"', rf'\1name = "{stem}"', source, count=1, flags=re.M)
    words = set(re.findall(r"\b\w+\b", source))
    needed = sorted(n for n, v in vars(module).items() if n in words and not n.startswith("__")
                    and n not in {new_cls, "annotations"} and not (isinstance(v, types.ModuleType)))
    modules = sorted(n for n, v in vars(module).items() if n in words and isinstance(v, types.ModuleType))
    lines = ['"""A copy of the %s indicator: change it and press Ctrl+S."""' % cls.name, ""]
    lines += [f"import {m.__name__} as {n}" if m.__name__.split(".")[-1] != n else
              (f"from {'.'.join(m.__name__.split('.')[:-1])} import {n}" if "." in m.__name__ else f"import {n}")
              for n in modules for m in [getattr(module, n)]]
    if needed:
        lines.append(f"from {cls.__module__} import {', '.join(needed)}")
    return script_path(stem), "\n".join(lines) + f"\n\n\n{source}"


def save(path: Path, source: str) -> None:
    """Write a script (atomically), creating the plugin folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(source, encoding="utf-8")
    tmp.replace(path)


@dataclass
class ReloadResult:
    ok: bool
    message: str
    line: int | None = None  #: the source line of an error
    classes: list[type[BaseIndicator]] = field(default_factory=list)  #: the indicators the script defines


def sample_ohlc(n: int = DRY_RUN_BARS) -> OHLC:
    from tradebot.evolve.primitives import _sample_ohlc

    return _sample_ohlc(n)


def _error_line(exc: BaseException, filename: str) -> int | None:
    if isinstance(exc, SyntaxError) and exc.filename == filename:
        return exc.lineno
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == filename:
            line = frame.lineno  # the innermost frame inside the script
    return line


def reload_script(path: Path, source: str, ohlc: OHLC | None = None) -> ReloadResult:
    """Compile, re-import and dry-run ``source`` as the plugin at ``path``; keep the old version on any error.

    1. ``compile()``: a ``SyntaxError`` stops here, with its line.
    2. The module is executed under ``tradebot_plugin_<stem>``; the registry replaces same-named classes.
    3. Every indicator the script defines must be screenable and survive ``compute`` on up to 500 bars
       (``ohlc``, default a synthetic series); a failure shows the line and restores the previous classes.
    """
    filename, name = str(path), module_name(path.stem)
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as exc:
        return ReloadResult(False, f"SyntaxError: {exc.msg} (line {exc.lineno})", exc.lineno)
    snapshot, old_module = dict(INDICATORS._items), sys.modules.get(name)
    module = types.ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module

    def restore(exc: BaseException, what: str) -> ReloadResult:
        INDICATORS._items.clear()
        INDICATORS._items.update(snapshot)
        if old_module is not None:
            sys.modules[name] = old_module
        else:
            sys.modules.pop(name, None)
        line = _error_line(exc, filename)
        where = f" (line {line})" if line else ""
        return ReloadResult(False, f"{what}: {type(exc).__name__}: {exc}{where}", line)

    try:
        with INDICATORS.reloading():
            exec(code, module.__dict__)
    except Exception as exc:
        return restore(exc, "Error while loading")
    classes = [c for c in INDICATORS if c.__module__ == name and snapshot.get(c.name) is not c]  # defined just now
    if not classes:
        return restore(ValueError("define a BaseIndicator subclass with a name"), "No indicator found")
    data = ohlc if ohlc is not None else sample_ohlc()
    data = OHLC(*(None if a is None else a[-DRY_RUN_BARS:] for a in data.fields()))
    for cls in classes:
        try:
            ok, why = is_screenable(cls)
            if not ok:
                raise ValueError(f"{cls.name} can't be scanned: {why}")
            with np.errstate(all="ignore"):
                out = cls.compute(data, **resolve_params(cls.params, {}))
            missing = [o for o in cls.outputs if o not in out]
            if missing:
                raise ValueError(f"compute() returned no {', '.join(missing)}")
            bad = [o for o in cls.outputs if len(out[o]) != len(data.close)]
            if bad:
                raise ValueError(f"{', '.join(bad)} must have one value per bar")
        except Exception as exc:
            return restore(exc, f"{cls.name} failed its dry run")
    plots = sum(len([o for o in c.outputs if o not in c.hidden and o not in c.alerts]) for c in classes)
    alerts = sum(len(c.alerts) for c in classes)
    names = ", ".join(c.name for c in classes)
    return ReloadResult(True, f"Loaded {names}: {_n(plots, 'plot')}, {_n(alerts, 'alert')}", None, classes)


def _n(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"
