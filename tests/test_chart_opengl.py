# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Optional OpenGL rendering of the chart canvas (#71): a setting, off by default, with a fallback.

CI and the offscreen tests have no GPU, so the GL viewport itself is only exercised where a context exists."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tradebot.config import Settings, load_config  # noqa: E402
from tradebot.gui import app as gui  # noqa: E402
from tradebot.gui import chart as chart_mod  # noqa: E402
from tradebot.gui.theme import ThemeManager  # noqa: E402

from .conftest import make_ohlc  # noqa: E402


@pytest.fixture
def themes(tmp_path):
    app = QApplication.instance() or QApplication([])  # keep a reference: a collected QApplication crashes Qt
    t = ThemeManager(user_dir=tmp_path / "themes")
    t.apply("dark")
    yield t
    del app


def test_off_by_default_uses_the_normal_viewport(themes):
    chart = chart_mod.ChartView(None, themes, ["TEST"])
    assert not chart.opengl and chart.opengl_note == ""
    assert type(chart.canvas.viewport()).__name__ == "QWidget"


def test_falls_back_and_says_why_when_opengl_is_unavailable(themes, monkeypatch):
    monkeypatch.setattr(chart_mod, "opengl_viewport", lambda: None)  # no driver, a VM, offscreen …
    chart = chart_mod.ChartView(None, themes, ["TEST"], opengl=True)
    assert not chart.opengl and "unavailable" in chart.opengl_note
    assert type(chart.canvas.viewport()).__name__ == "QWidget"
    chart.df = make_ohlc(300)
    chart.redraw(autorange=True)  # and the chart still draws
    chart.resize(800, 500)
    assert not chart.canvas.grab().isNull()


def test_the_real_probe_never_raises(themes):
    """Whatever this machine has, asking for GPU rendering gives a working chart: OpenGL or the fallback."""
    chart = chart_mod.ChartView(None, themes, ["TEST"], opengl=True)
    assert chart.opengl != bool(chart.opengl_note)
    if chart.opengl:
        assert type(chart.canvas.viewport()).__name__ == "QOpenGLWidget"


def test_main_window_shows_the_fallback_once_in_the_status_bar(themes, tmp_path, monkeypatch):
    monkeypatch.setattr(gui.ChartView, "load", lambda self: None)  # no network
    monkeypatch.setattr(chart_mod, "opengl_viewport", lambda: None)
    window = gui.MainWindow(themes, "dark", Settings(tmp_path / "settings.json"), opengl=True)
    assert "GPU rendering unavailable" in window.statusBar().currentMessage()
    window.close()


def test_setting_is_saved_in_the_config(themes, tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "config.toml"))
    from tradebot.gui.settings import SettingsDialog

    dialog = SettingsDialog(themes, "dark")
    messages = []
    dialog.message.connect(messages.append)
    assert not dialog.gpu_box.isChecked()  # off by default
    dialog.gpu_box.setChecked(True)
    assert load_config()["gpu_rendering"] is True and "restart" in messages[-1]
    assert SettingsDialog(themes, "dark").gpu_box.isChecked()  # shown as saved next time
    dialog.gpu_box.setChecked(False)
    assert load_config()["gpu_rendering"] is False
