# SPDX-License-Identifier: GPL-3.0-or-later
"""App-launcher entry on Linux, Windows and macOS (#213): `tradebot desktop install|uninstall [--system] [--dev]`.

* Linux: a .desktop file, icons and AppStream metadata under ~/.local/share (or /usr/local/share).
* Windows: a Start Menu shortcut (.lnk) made with PowerShell's WScript.Shell, no extra dependency.
* macOS: the release .app copied into ~/Applications (/Applications), or a small launcher .app for pip/uv installs
  and source checkouts.
Works for the PyInstaller build (sys.frozen), a pip/uv install and a source checkout."""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).with_name("gui") / "assets" / "desktop"
ICON_SIZES = (32, 48, 64, 128, 256, 512)
APP_ID = "io.github.damarxrider.tradebot"
RESERVED = set(" \t\n\"'\\><~|&;$*?#()`")


@dataclass(frozen=True)
class Variant:
    desktop_id: str  # Linux: file stem of the .desktop file and the icon name
    name: str  # shown in the launcher; also the .lnk / .app name
    dev: bool


RELEASE = Variant("tradebot", "Tradebot", False)
DEV = Variant("tradebot-dev", "Tradebot (dev)", True)


def current_os() -> str:
    p = sys.platform
    if p.startswith("linux"):
        return "linux"
    if p == "win32":
        return "windows"
    if p == "darwin":
        return "macos"
    raise RuntimeError(f"app-launcher entries are not supported on {p}")


def quote_arg(arg: str) -> str:
    """One Exec argument per the Desktop Entry spec: quoted when it holds reserved characters, `"` `` ` `` `$` `\\`
    escaped, `%` doubled (field codes). Backslashes are doubled once more for the file's own string escaping."""
    arg = arg.replace("%", "%%")
    if any(c in RESERVED for c in arg):
        for c in '\\"`$':
            arg = arg.replace(c, "\\" + c)
        arg = f'"{arg}"'
    return arg.replace("\\", "\\\\")


def launcher_command() -> list[str]:
    """The absolute command that starts this tradebot: the frozen binary, the running `tradebot` script (pip/uv
    install, source checkout's venv), or the interpreter running the CLI module."""
    if getattr(sys, "frozen", False):
        return [str(Path(sys.executable).resolve())]
    script = Path(sys.argv[0])
    if script.stem == "tradebot" and script.is_file():
        return [str(script.absolute())]
    found = shutil.which("tradebot")
    if found:
        return [str(Path(found).absolute())]
    return [sys.executable, "-m", "tradebot.cli.main"]


# ---------------------------------------------------------------- Linux

def exec_line(variant: Variant, command: list[str]) -> str:
    argv = [*command, "gui"]
    if variant.dev:  # the GUI reads the id to name its window, so the dev entry does not group with the release one
        argv = ["env", f"TRADEBOT_DESKTOP_ID={variant.desktop_id}", *argv]
    return " ".join(quote_arg(a) for a in argv)


def data_dir(system: bool = False, prefix: Path | None = None) -> Path:
    if prefix is not None:
        return Path(prefix) / "share"
    if system:
        return Path("/usr/local/share")
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def desktop_text(variant: Variant, command: list[str]) -> str:
    text = (ASSETS / "tradebot.desktop.in").read_text(encoding="utf-8")
    return (text.replace("@NAME@", variant.name).replace("@EXEC@", exec_line(variant, command))
            .replace("@ICON@", variant.desktop_id).replace("@WMCLASS@", variant.desktop_id if variant.dev else "tradebot"))


def targets(variant: Variant, base: Path) -> dict[Path, Path | None]:
    """Every Linux file install writes (and uninstall removes) -> its source in the package (None: generated)."""
    out: dict[Path, Path | None] = {base / "applications" / f"{variant.desktop_id}.desktop": None}
    for size in ICON_SIZES:
        out[base / "icons" / "hicolor" / f"{size}x{size}" / "apps" / f"{variant.desktop_id}.png"] = ASSETS / f"tradebot-{size}.png"
    out[base / "icons" / "hicolor" / "scalable" / "apps" / f"{variant.desktop_id}.svg"] = ASSETS / "tradebot.svg"
    if not variant.dev:
        out[base / "metainfo" / f"{APP_ID}.metainfo.xml"] = ASSETS / f"{APP_ID}.metainfo.xml"
    return out


def refresh_caches(base: Path) -> None:
    """Best effort: a missing tool or a failing refresh only means the menu updates a little later."""
    for cmd in (["update-desktop-database", "-q", str(base / "applications")],
                ["gtk-update-icon-cache", "-q", "-t", "-f", str(base / "icons" / "hicolor")]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, check=False, timeout=30, capture_output=True)
            except (OSError, subprocess.SubprocessError):
                pass


def _install_linux(variant, system, prefix, dry_run, command) -> list[Path]:
    base = data_dir(system, prefix)
    if system and prefix is None and not dry_run and os.geteuid() != 0:
        raise RuntimeError(f"--system writes to {base}, which needs root: run it with sudo (or leave out --system)")
    files = targets(variant, base)
    if not dry_run:
        for dest, src in files.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src is None:
                dest.write_text(desktop_text(variant, command), encoding="utf-8")
            else:
                shutil.copyfile(src, dest)
            dest.chmod(0o644)
        if prefix is None:
            refresh_caches(base)
    return list(files)


def _uninstall_linux(variant, system, prefix, dry_run) -> list[Path]:
    base = data_dir(system, prefix)
    removed = [p for p in targets(variant, base) if p.exists()]
    if not dry_run:
        for p in removed:
            p.unlink()
        if prefix is None and removed:
            refresh_caches(base)
    return removed


# ---------------------------------------------------------------- Windows

def start_menu_dir(system: bool = False, prefix: Path | None = None) -> Path:
    if prefix is not None:
        return Path(prefix)
    root = os.environ.get("PROGRAMDATA") if system else os.environ.get("APPDATA")
    if not root:
        raise RuntimeError("%PROGRAMDATA%" if system else "%APPDATA% is not set")
    return Path(root) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def shortcut_target(command: list[str]) -> tuple[str, str]:
    """(program, arguments) for the shortcut: the windowed tradebot-gui.exe next to the frozen tradebot.exe, else the
    interpreter's pythonw.exe (no console window) running the CLI module."""
    if getattr(sys, "frozen", False):
        return str(Path(command[0]).with_name("tradebot-gui.exe")), ""
    if len(command) > 1:  # [python, -m, module]: command[0] is the interpreter
        py = Path(command[0])
    else:  # a Scripts/tradebot.exe launcher: its virtualenv's interpreter is next to it
        py = Path(command[0]).with_name("python.exe")
    pythonw = py.with_name("pythonw.exe")
    return str(pythonw if pythonw.exists() else py), "-m tradebot.cli.main gui"


def write_shortcut(lnk: Path, target: str, arguments: str, icon: str) -> None:
    """Create the .lnk with PowerShell's WScript.Shell. Values travel in environment variables, so no quoting."""
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if ps is None:
        raise RuntimeError("PowerShell not found, cannot create the Start Menu shortcut")
    script = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:TB_LNK); $s.TargetPath = $env:TB_TARGET; "
              "$s.Arguments = $env:TB_ARGS; $s.WorkingDirectory = Split-Path $env:TB_TARGET; $s.IconLocation = $env:TB_ICON; "
              "$s.Description = 'Charts, backtests, optimization and trading bots'; $s.Save()")
    env = {**os.environ, "TB_LNK": str(lnk), "TB_TARGET": target, "TB_ARGS": arguments, "TB_ICON": icon}
    done = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", script], env=env, capture_output=True,
                          text=True, timeout=60)
    if done.returncode != 0 or not lnk.exists():
        raise RuntimeError(f"creating {lnk} failed: {(done.stderr or done.stdout).strip()}")


def _install_windows(variant, system, prefix, dry_run, command) -> list[Path]:
    lnk = start_menu_dir(system, prefix) / f"{variant.name}.lnk"
    if not dry_run:
        lnk.parent.mkdir(parents=True, exist_ok=True)
        target, arguments = shortcut_target(command)
        write_shortcut(lnk, target, arguments, str(ASSETS / "tradebot.ico"))
    return [lnk]


def _uninstall_windows(variant, system, prefix, dry_run) -> list[Path]:
    lnk = start_menu_dir(system, prefix) / f"{variant.name}.lnk"
    if not lnk.exists():
        return []
    if not dry_run:
        lnk.unlink()
    return [lnk]


# ---------------------------------------------------------------- macOS

def applications_dir(system: bool = False, prefix: Path | None = None) -> Path:
    if prefix is not None:
        return Path(prefix) / "Applications"
    return Path("/Applications") if system else Path.home() / "Applications"


def running_bundle() -> Path | None:
    """The tradebot.app this frozen binary runs from (Contents/MacOS/<exe>), or None."""
    if not getattr(sys, "frozen", False):
        return None
    exe = Path(sys.executable).resolve()
    return exe.parents[2] if len(exe.parents) > 2 and exe.parents[2].suffix == ".app" else None


def write_stub_app(app: Path, variant: Variant, command: list[str]) -> None:
    """A minimal .app whose launcher script runs `<command> gui` (pip/uv installs, source checkouts)."""
    shutil.rmtree(app, ignore_errors=True)
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Resources").mkdir()
    shutil.copyfile(ASSETS / "tradebot.icns", app / "Contents" / "Resources" / "tradebot.icns")
    plist = {"CFBundleName": variant.name, "CFBundleDisplayName": variant.name, "CFBundleExecutable": "launch",
             "CFBundleIdentifier": APP_ID + (".dev" if variant.dev else ".launcher"), "CFBundleIconFile": "tradebot",
             "CFBundlePackageType": "APPL", "CFBundleVersion": "1", "LSApplicationCategoryType": "public.app-category.finance",
             "NSHighResolutionCapable": True}
    (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps(plist))
    launcher = app / "Contents" / "MacOS" / "launch"
    launcher.write_text("#!/bin/sh\nexec " + " ".join(shlex.quote(a) for a in [*command, "gui"]) + "\n", encoding="utf-8")
    launcher.chmod(0o755)


def _install_macos(variant, system, prefix, dry_run, command) -> list[Path]:
    app = applications_dir(system, prefix) / f"{variant.name}.app"
    bundle = None if variant.dev else running_bundle()
    if not dry_run:
        app.parent.mkdir(parents=True, exist_ok=True)
        if bundle is not None:
            if bundle.resolve() != app.resolve():  # a copy keeps the ad-hoc signature (symlinks=True keeps the bundle's links)
                shutil.rmtree(app, ignore_errors=True)
                shutil.copytree(bundle, app, symlinks=True)
        else:
            write_stub_app(app, variant, command)
    return [app]


def _uninstall_macos(variant, system, prefix, dry_run) -> list[Path]:
    app = applications_dir(system, prefix) / f"{variant.name}.app"
    if not app.exists():
        return []
    if not dry_run:
        shutil.rmtree(app)
    return [app]


# ---------------------------------------------------------------- public

def install(*, dev: bool = False, system: bool = False, prefix: Path | None = None, dry_run: bool = False,
            command: list[str] | None = None) -> list[Path]:
    """Write the entry; returns what was written. Overwrites its own earlier files, so running it twice is harmless."""
    system_os = current_os()
    command = command or launcher_command()
    fn = {"linux": _install_linux, "windows": _install_windows, "macos": _install_macos}[system_os]
    try:
        return fn(DEV if dev else RELEASE, system, prefix, dry_run, command)
    except PermissionError as exc:
        raise RuntimeError(f"no permission to write {exc.filename}: --system needs administrator rights") from exc


def uninstall(*, dev: bool = False, system: bool = False, prefix: Path | None = None, dry_run: bool = False) -> list[Path]:
    """Remove exactly what install wrote; returns what existed."""
    fn = {"linux": _uninstall_linux, "windows": _uninstall_windows, "macos": _uninstall_macos}[current_os()]
    try:
        return fn(DEV if dev else RELEASE, system, prefix, dry_run)
    except PermissionError as exc:
        raise RuntimeError(f"no permission to remove {exc.filename}: --system needs administrator rights") from exc
