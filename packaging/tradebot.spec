# SPDX-License-Identifier: GPL-3.0-or-later
# PyInstaller build of the tradebot CLI + GUI as a one-dir bundle in dist/tradebot/.
# On Windows it also contains tradebot-gui.exe, which opens the GUI without a console window;
# on macOS the bundle is wrapped in dist/tradebot.app, whose launcher opens the GUI.
#   uv sync --extra gui --group build && uv run pyinstaller packaging/tradebot.spec
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# Every tradebot module, so directory plugins can import any of them; plus data files
# (certificates, exchange/timezone tables) that ccxt and yfinance read at runtime, and
# plotext's C++ kernel, which it loads via ctypes for the terminal charts.
hiddenimports = collect_submodules("tradebot")
datas = collect_data_files("ccxt") + collect_data_files("yfinance")
binaries = collect_dynamic_libs("plotext", search_patterns=["*.so", "*.dll", "*.pyd"])

options = dict(
    binaries=binaries,
    hiddenimports=hiddenimports,
    datas=datas,
    excludes=["torch", "tkinter"],  # CPU-only build; the GPU optimizer falls back to NumPy
)
a = Analysis(["entry.py"], **options)
exe = EXE(PYZ(a.pure), a.scripts, [], exclude_binaries=True, name="tradebot", console=True)
parts = [exe, a.binaries, a.datas]

if sys.platform in ("win32", "darwin"):
    g = Analysis(["gui_entry.py"], **options)
    gui_exe = EXE(PYZ(g.pure), g.scripts, [], exclude_binaries=True, name="tradebot-gui", console=False)
    parts += [gui_exe, g.binaries, g.datas]  # COLLECT drops the files both builds share

coll = COLLECT(*parts, name="tradebot")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="tradebot.app",
        bundle_identifier="io.github.damarxrider.tradebot",
        info_plist={"CFBundleExecutable": "tradebot-gui", "NSHighResolutionCapable": True},  # opening the app starts the GUI
    )
