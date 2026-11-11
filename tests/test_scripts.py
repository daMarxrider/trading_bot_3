# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Scanner scripts (#115): the reload keeps the last good version when a script breaks."""

import logging
import sys

import pytest

from tradebot import scripts
from tradebot.indicators import INDICATORS

GOOD = '''
from tradebot.core import ops
from tradebot.core.registry import IntParam
from tradebot.indicators.base import BaseIndicator

class Mine(BaseIndicator):
    name = "live_test_ind"
    params = {"period": IntParam(5, 2, 50)}
    outputs = ("avg", "up")
    alerts = {"up": "close above"}

    @classmethod
    def compute(cls, o, period):
        avg = ops.sma(o.close, period) * %s
        return {"avg": avg, "up": ops.flag(o.close > avg, avg)}
'''


@pytest.fixture
def plugin_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEBOT_PLUGIN_DIR", str(tmp_path))
    yield tmp_path
    INDICATORS._items.pop("live_test_ind", None)
    INDICATORS._items.pop("renamed_test_ind", None)
    for name in [m for m in sys.modules if m.startswith("tradebot_plugin_live") or m.startswith("tradebot_plugin_my_ind")]:
        sys.modules.pop(name)


def test_reload_replaces_the_registered_class_without_a_warning(plugin_dir, caplog):
    path = plugin_dir / "live.py"
    first = scripts.reload_script(path, GOOD % "1.0")
    assert first.ok and first.message == "Loaded live_test_ind: 1 plot, 1 alert"
    old = INDICATORS.get("live_test_ind")
    with caplog.at_level(logging.WARNING):
        second = scripts.reload_script(path, GOOD % "2.0")
    assert second.ok and INDICATORS.get("live_test_ind") is not old and not caplog.records
    assert INDICATORS.get("live_test_ind") is second.classes[0]


def test_a_syntax_error_keeps_the_old_class_and_names_the_line(plugin_dir):
    path = plugin_dir / "live.py"
    scripts.reload_script(path, GOOD % "1.0")
    good = INDICATORS.get("live_test_ind")
    res = scripts.reload_script(path, GOOD % "1.0" + "\ndef broken(:\n    pass\n")
    assert not res.ok and res.line and "SyntaxError" in res.message
    assert INDICATORS.get("live_test_ind") is good


def test_runtime_errors_in_import_and_compute_restore_the_old_class(plugin_dir):
    path = plugin_dir / "live.py"
    scripts.reload_script(path, GOOD % "1.0")
    good = INDICATORS.get("live_test_ind")
    at_import = scripts.reload_script(path, GOOD % "1.0" + "\nraise RuntimeError('boom')\n")
    assert not at_import.ok and "boom" in at_import.message and at_import.line
    assert INDICATORS.get("live_test_ind") is good
    bad_compute = scripts.reload_script(path, (GOOD % "1.0").replace("ops.sma(o.close, period) * 1.0", "o.close / 0 + nope"))
    assert not bad_compute.ok and "dry run" in bad_compute.message and bad_compute.line
    assert INDICATORS.get("live_test_ind") is good and sys.modules["tradebot_plugin_live"].__dict__["Mine"] is good


def test_missing_outputs_and_unscreenable_scripts_are_rejected(plugin_dir):
    path = plugin_dir / "live.py"
    scripts.reload_script(path, GOOD % "1.0")
    good = INDICATORS.get("live_test_ind")
    short = (GOOD % "1.0").replace('return {"avg": avg, "up"', 'return {"avg": avg, "xx"')
    assert not scripts.reload_script(path, short).ok
    nothing = (GOOD % "1.0").replace('alerts = {"up": "close above"}', 'hidden = ("avg", "up")')
    res = scripts.reload_script(path, nothing)
    assert not res.ok and "can't be scanned" in res.message
    assert not scripts.reload_script(path, "x = 1\n").ok
    assert INDICATORS.get("live_test_ind") is good


def test_renaming_registers_a_new_indicator_and_keeps_the_old_one(plugin_dir):
    path = plugin_dir / "live.py"
    scripts.reload_script(path, GOOD % "1.0")
    assert scripts.reload_script(path, (GOOD % "1.0").replace("live_test_ind", "renamed_test_ind")).ok
    assert "renamed_test_ind" in INDICATORS and "live_test_ind" in INDICATORS


def test_the_new_script_template_loads_and_scans(plugin_dir):
    path, source = scripts.new_script()
    assert path.parent == plugin_dir and path.name == "my_indicator.py"
    res = scripts.reload_script(path, source)
    assert res.ok and res.message == "Loaded my_indicator: 1 plot, 1 alert", res.message
    cls = res.classes[0]
    assert cls.name == "my_indicator" and list(cls.alerts) == ["above"] and list(cls.params) == ["period"]
    scripts.save(path, source)
    assert scripts.new_script()[0].name == "my_indicator_2.py"  # taken names are skipped
    assert scripts.is_script(cls) and scripts.source_path(cls) == path
    INDICATORS._items.pop("my_indicator")


def test_copy_to_my_scripts_makes_a_working_renamed_copy(plugin_dir):
    for name in ("rsi", "ma_cross", "ssw", "macd"):  # SSW uses another class, MA Cross a module helper
        path, source = scripts.copy_of(INDICATORS.get(name))
        assert f'name = "{name}_copy"' in source
        res = scripts.reload_script(path, source)
        assert res.ok, (name, res.message)
        assert INDICATORS.get(name) is not res.classes[0] and f"{name}_copy" in INDICATORS
        INDICATORS._items.pop(f"{name}_copy")
        sys.modules.pop(f"tradebot_plugin_{name}_copy", None)


def test_a_real_series_can_be_used_for_the_dry_run(plugin_dir):
    from tradebot.core.series import OHLC
    import numpy as np

    flat = OHLC(*(np.full(30, 5.0) for _ in range(4)))
    assert scripts.reload_script(plugin_dir / "live.py", GOOD % "1.0", flat).ok
