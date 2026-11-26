# SPDX-License-Identifier: GPL-3.0-or-later
# PyInstaller build of the tradebot CLI + GUI as a one-dir bundle in dist/tradebot/.
# On Windows it also contains tradebot-gui.exe, which opens the GUI without a console window;
# on macOS the bundle is wrapped in dist/tradebot.app, whose launcher opens the GUI.
#   uv sync --extra gui --group build && uv run pyinstaller packaging/tradebot.spec
import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# Every tradebot module, so directory plugins can import any of them; plus data files
# (certificates, exchange/timezone tables) that ccxt and yfinance read at runtime, and
# plotext's C++ kernel, which it loads via ctypes for the terminal charts.
hiddenimports = collect_submodules("tradebot")
datas = collect_data_files("ccxt") + collect_data_files("yfinance")
# tradebot's licence and those of every bundled library go into the bundle (#58).
sys.path.insert(0, SPECPATH)
import third_party_notices  # noqa: E402

notices = os.path.join(workpath, "THIRD_PARTY_NOTICES.txt")
os.makedirs(workpath, exist_ok=True)
with open(notices, "w", encoding="utf-8") as fh:
    fh.write(third_party_notices.render())
datas += [(os.path.join(SPECPATH, "..", "LICENSE"), "."), (notices, ".")]
# the UI fonts and drawn icons (#191)
datas += [(os.path.join(SPECPATH, "..", "src", "tradebot", "gui", "assets"), os.path.join("tradebot", "gui", "assets"))]
# the launcher entry's icons (#213): the .ico is also the exe's icon on Windows, the .icns the .app's on macOS
ICONS = os.path.join(SPECPATH, "..", "src", "tradebot", "gui", "assets", "desktop")
# The version comes from tradebot/__init__.py, which release.yml stamps from the tag. Read, not imported: importing
# tradebot pulls in every indicator and broker.
with open(os.path.join(SPECPATH, "..", "src", "tradebot", "__init__.py"), encoding="utf-8") as fh:
    VERSION = re.search(r'^__version__ = "([^"]+)"', fh.read(), re.M).group(1)


def win_version_info(exe_name, description):
    """The exe's Windows version resource (Properties > Details). SignPath signing requires the product name to be
    the project name and the product version to be the same in every build of a release."""
    from PyInstaller.utils.win32.versioninfo import (FixedFileInfo, StringFileInfo, StringStruct, StringTable,
                                                     VarFileInfo, VarStruct, VSVersionInfo)

    numbers = (tuple(int(n) for n in re.findall(r"\d+", VERSION)[:4]) + (0, 0, 0, 0))[:4]
    strings = {
        "CompanyName": "daMarxrider",
        "FileDescription": description,
        "FileVersion": VERSION,
        "InternalName": exe_name,
        "LegalCopyright": "Copyright (C) 2026 daMarxrider. GPL-3.0-or-later",
        "OriginalFilename": f"{exe_name}.exe",
        "ProductName": "tradebot",
        "ProductVersion": VERSION,
    }
    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=numbers, prodvers=numbers),
        kids=[
            StringFileInfo([StringTable("040904B0", [StringStruct(k, v) for k, v in strings.items()])]),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),  # US English, Unicode
        ],
    )


def win_exe_options(exe_name, description):
    if sys.platform != "win32":
        return {}
    return {"icon": os.path.join(ICONS, "tradebot.ico"), "version": win_version_info(exe_name, description)}


binaries = collect_dynamic_libs("plotext", search_patterns=["*.so", "*.dll", "*.pyd"])

options = dict(
    binaries=binaries,
    hiddenimports=hiddenimports,
    datas=datas,
    # CPU-only build (the GPU optimizer falls back to NumPy). Third-party tools tradebot integrates, such as
    # pytr or the Scalable CLI, are never bundled: users install them themselves (#58).
    excludes=["torch", "tkinter", "pytr"],
)
a = Analysis(["entry.py"], **options)
exe = EXE(PYZ(a.pure), a.scripts, [], exclude_binaries=True, name="tradebot", console=True,
          **win_exe_options("tradebot", "tradebot command line"))
parts = [exe, a.binaries, a.datas]

if sys.platform in ("win32", "darwin"):
    g = Analysis(["gui_entry.py"], **options)
    gui_exe = EXE(PYZ(g.pure), g.scripts, [], exclude_binaries=True, name="tradebot-gui",
                  console=False, **win_exe_options("tradebot-gui", "tradebot desktop app"))
    parts += [gui_exe, g.binaries, g.datas]  # COLLECT drops the files both builds share

coll = COLLECT(*parts, name="tradebot")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="tradebot.app",
        bundle_identifier="io.github.damarxrider.tradebot",
        icon=os.path.join(ICONS, "tradebot.icns"),
        info_plist={
            "CFBundleExecutable": "tradebot-gui",  # opening the app starts the GUI
            "CFBundleName": "Tradebot",
            "CFBundleDisplayName": "Tradebot",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSApplicationCategoryType": "public.app-category.finance",
            "NSHighResolutionCapable": True,
        },
    )
