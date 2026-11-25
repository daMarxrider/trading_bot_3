# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Optional real backdrop blur for the Glass theme (#7).

The compositor blurs whatever is behind the window; the app only has to make its own background
translucent and ask for it. Support differs per platform, so :func:`detect_backend` names the way
that applies and returns ``"none"`` whenever there is none, in which case the painted gradient of
:class:`tradebot.gui.theme.GlassBackdrop` stays. Nothing here may raise: every native call is
wrapped and a failure just means "no blur".

Backends:

* ``x11``: the ``_KDE_NET_WM_BLUR_BEHIND_REGION`` window property, honoured by KWin (also for
  XWayland windows) and by picom-style compositors. Needs Qt's ``xcb`` platform.
* ``windows``: the DWM acrylic backdrop (Windows 11 22H2, build 22621 or newer).
* ``macos``: an ``NSVisualEffectView`` behind the content (needs ``pyobjc``).
* ``none``: native Wayland (Qt has no binding for ``org_kde_kwin_blur`` or ``ext-background-effect``;
  wlroots/Hyprland blur is a compositor-side rule), the offscreen platform, anything else.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

log = logging.getLogger(__name__)

WIN_ACRYLIC_BUILD = 22621
_BLUR_PROPERTY = b"_KDE_NET_WM_BLUR_BEHIND_REGION"
_NO_DISPLAY_PLATFORMS = ("offscreen", "minimal", "vnc", "eglfs", "linuxfb")


def detect_backend(qt_platform: str, sys_platform: str | None = None, env=None, win_build: int | None = None) -> str:
    """``"x11"``, ``"windows"``, ``"macos"`` or ``"none"`` for this platform. Pure, so it's testable."""
    sys_platform = sys.platform if sys_platform is None else sys_platform
    env = os.environ if env is None else env
    qt_platform = (qt_platform or "").lower()
    if qt_platform in _NO_DISPLAY_PLATFORMS:
        return "none"
    if sys_platform.startswith("win"):
        if win_build is None:
            win_build = _windows_build()
        return "windows" if qt_platform.startswith("windows") and win_build >= WIN_ACRYLIC_BUILD else "none"
    if sys_platform == "darwin":
        return "macos" if qt_platform == "cocoa" and _has_pyobjc() else "none"
    if qt_platform.startswith("xcb") and env.get("DISPLAY"):
        return "x11"
    return "none"


def _windows_build() -> int:
    try:
        return int(sys.getwindowsversion().build)  # type: ignore[attr-defined]
    except Exception:
        return 0


def _has_pyobjc() -> bool:
    try:
        import AppKit  # noqa: F401
        import objc  # noqa: F401
        return True
    except Exception:
        return False


def current_backend() -> str:
    from PyQt6.QtGui import QGuiApplication
    return detect_backend(QGuiApplication.platformName())


def wants_blur(config: dict) -> bool:
    """``glass_blur`` in the config file: off by default, ``true`` to try it."""
    value = config.get("glass_blur", False)
    return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes", "auto")


def prepare(window: QWidget, backend: str) -> bool:
    """Before the window is first shown: give it an alpha channel when a backend exists."""
    if backend == "none":
        return False
    window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    return True


def set_blur(window: QWidget, backend: str, on: bool) -> bool:
    """Ask the compositor to blur behind ``window`` (or stop). True when blur is now in effect."""
    if backend == "none":
        return False
    try:
        wid = int(window.winId())
        ok = {"x11": _x11, "windows": _windows, "macos": _macos}[backend](wid, on)
    except Exception:
        log.debug("backdrop blur (%s) failed", backend, exc_info=True)
        return False
    return bool(ok) and on


def _x11(wid: int, on: bool) -> bool:
    xcb = ctypes.CDLL("libxcb.so.1")
    libc = ctypes.CDLL(None)
    vp, u32 = ctypes.c_void_p, ctypes.c_uint32
    xcb.xcb_connect.restype = vp
    xcb.xcb_connect.argtypes = [ctypes.c_char_p, vp]
    xcb.xcb_connection_has_error.argtypes = [vp]
    xcb.xcb_intern_atom.restype = u32
    xcb.xcb_intern_atom.argtypes = [vp, ctypes.c_uint8, ctypes.c_uint16, ctypes.c_char_p]
    xcb.xcb_intern_atom_reply.restype = vp
    xcb.xcb_intern_atom_reply.argtypes = [vp, u32, vp]
    xcb.xcb_change_property.argtypes = [vp, ctypes.c_uint8, u32, u32, u32, ctypes.c_uint8, u32, vp]
    xcb.xcb_delete_property.argtypes = [vp, u32, u32]
    xcb.xcb_flush.argtypes = [vp]
    xcb.xcb_disconnect.argtypes = [vp]
    conn = xcb.xcb_connect(None, None)
    try:
        if not conn or xcb.xcb_connection_has_error(conn):
            return False
        name = _BLUR_PROPERTY
        reply = xcb.xcb_intern_atom_reply(conn, xcb.xcb_intern_atom(conn, 0, len(name), name), None)
        if not reply:
            return False
        prop = ctypes.cast(reply, ctypes.POINTER(u32))[2]  # the atom follows 8 header bytes
        libc.free(reply)
        if on:
            region = (u32 * 4)(0, 0, 0, 0)  # an empty rectangle means the whole window
            xcb.xcb_change_property(conn, 0, wid, prop, 6, 32, 4, ctypes.cast(region, vp))  # 6 = CARDINAL
        else:
            xcb.xcb_delete_property(conn, wid, prop)
        xcb.xcb_flush(conn)
        return True
    finally:
        if conn:
            xcb.xcb_disconnect(conn)


def _windows(wid: int, on: bool) -> bool:
    dwm = ctypes.windll.dwmapi  # type: ignore[attr-defined]
    backdrop = ctypes.c_int(3 if on else 1)  # DWMSBT_TRANSIENTWINDOW (acrylic) / DWMSBT_NONE
    return dwm.DwmSetWindowAttribute(ctypes.c_void_p(wid), 38, ctypes.byref(backdrop), 4) == 0  # 38 = SYSTEMBACKDROP_TYPE


def _macos(wid: int, on: bool) -> bool:
    import AppKit
    import objc
    view = objc.objc_object(c_void_p=ctypes.c_void_p(wid))
    win = view.window()
    content = win.contentView()
    for sub in list(content.subviews()):
        if sub.className() == "NSVisualEffectView":
            sub.removeFromSuperview()
    win.setOpaque_(not on)
    win.setBackgroundColor_(AppKit.NSColor.clearColor() if on else AppKit.NSColor.windowBackgroundColor())
    if on:
        fx = AppKit.NSVisualEffectView.alloc().initWithFrame_(content.bounds())
        fx.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        fx.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
        fx.setState_(AppKit.NSVisualEffectStateActive)
        fx.setMaterial_(AppKit.NSVisualEffectMaterialUnderWindowBackground)
        content.addSubview_positioned_relativeTo_(fx, AppKit.NSWindowBelow, None)
    return True
