# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Glass backdrop blur (#7): capability detection and the fallback to the painted gradient."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from tradebot.gui import blur  # noqa: E402
from tradebot.gui.theme import GlassBackdrop, ThemeManager  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_detect_backend():
    d = blur.detect_backend
    assert d("offscreen", "linux", {"DISPLAY": ":0"}) == "none"
    assert d("xcb", "linux", {"DISPLAY": ":0"}) == "x11"
    assert d("xcb", "linux", {}) == "none"
    assert d("wayland", "linux", {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}) == "none"
    assert d("windows", "win32", {}, win_build=22631) == "windows"
    assert d("windows", "win32", {}, win_build=19045) == "none"
    assert d("cocoa", "darwin", {}) in ("macos", "none")  # depends on pyobjc being installed


def test_wants_blur():
    assert not blur.wants_blur({})
    assert blur.wants_blur({"glass_blur": True})
    assert not blur.wants_blur({"glass_blur": False})
    assert blur.wants_blur({"glass_blur": "on"})


def test_no_backend_is_a_noop(app):
    w = QWidget()
    try:
        assert blur.prepare(w, "none") is False
        assert blur.set_blur(w, "none", True) is False
        assert not w.testAttribute(blur.Qt.WidgetAttribute.WA_TranslucentBackground)
    finally:
        w.close()
        w.deleteLater()


def test_failing_backend_never_raises(app, monkeypatch):
    def boom(wid, on):
        raise OSError("no libxcb")

    monkeypatch.setattr(blur, "_x11", boom)
    w = QWidget()
    try:
        assert blur.set_blur(w, "x11", True) is False
    finally:
        w.close()
        w.deleteLater()


def test_offscreen_keeps_gradient(app):
    from tradebot.gui.app import MainWindow
    win = MainWindow(ThemeManager(), "glass", glass_blur=True)
    try:
        assert win.blur_backend == "none"
        win.show()
        assert win.backdrop.blur_active is False
        assert not win.testAttribute(blur.Qt.WidgetAttribute.WA_TranslucentBackground)
    finally:
        win.close()
        win.deleteLater()


def test_backdrop_paints_with_blur_flag(app):
    themes = ThemeManager()
    themes.apply("glass")
    b = GlassBackdrop(themes)
    try:
        b.resize(300, 200)
        opaque = b.grab().toImage().pixelColor(5, 5).alpha()
        b.blur_active = True
        translucent = b.grab().toImage().pixelColor(5, 5).alpha()
        assert opaque == 255
        assert translucent < 255
    finally:
        b.close()
        b.deleteLater()
