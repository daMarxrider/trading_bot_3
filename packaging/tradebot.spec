# SPDX-License-Identifier: GPL-3.0-or-later
# PyInstaller build of the tradebot CLI + GUI as a one-dir bundle in dist/tradebot/.
#   uv sync --extra gui --group build && uv run pyinstaller packaging/tradebot.spec
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# Every tradebot module, so directory plugins can import any of them; plus data files
# (certificates, exchange/timezone tables) that ccxt and yfinance read at runtime, and
# plotext's C++ kernel, which it loads via ctypes for the terminal charts.
hiddenimports = collect_submodules("tradebot")
datas = collect_data_files("ccxt") + collect_data_files("yfinance")
binaries = collect_dynamic_libs("plotext", search_patterns=["*.so", "*.dll", "*.pyd"])

a = Analysis(
    ["entry.py"],
    binaries=binaries,
    hiddenimports=hiddenimports,
    datas=datas,
    excludes=["torch", "tkinter"],  # CPU-only build; the GPU optimizer falls back to NumPy
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="tradebot", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="tradebot")
