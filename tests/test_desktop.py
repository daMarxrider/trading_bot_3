# SPDX-License-Identifier: GPL-3.0-or-later
"""Linux app-launcher entry (#213). Everything goes into tmp_path (XDG_DATA_HOME or --prefix), never the real home."""

import shutil
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from tradebot import desktop
from tradebot.cli.main import app

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
runner = CliRunner()


def exec_of(path):
    return next(line for line in path.read_text().splitlines() if line.startswith("Exec=")).removeprefix("Exec=")


@pytest.fixture(autouse=True)
def private_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(desktop, "refresh_caches", lambda base: None)


@linux_only
def test_install_writes_desktop_file_icons_and_metainfo(tmp_path):
    files = desktop.install(command=["/opt/tb/tradebot"])
    base = tmp_path / "xdg"
    assert all(f.is_file() and base in f.parents for f in files)
    text = (base / "applications" / "tradebot.desktop").read_text()
    assert "Name=Tradebot\n" in text and "Icon=tradebot\n" in text and "Terminal=false" in text
    assert exec_of(base / "applications" / "tradebot.desktop") == "/opt/tb/tradebot gui"
    for size in desktop.ICON_SIZES:
        assert (base / "icons" / "hicolor" / f"{size}x{size}" / "apps" / "tradebot.png").is_file()
    assert (base / "metainfo" / f"{desktop.APP_ID}.metainfo.xml").is_file()
    assert not (tmp_path / "home").exists()


@linux_only
def test_install_is_idempotent_and_uninstall_removes_exactly_what_was_written(tmp_path):
    first = desktop.install(command=["/x/tradebot"])
    again = desktop.install(command=["/x/tradebot"])
    assert first == again
    removed = desktop.uninstall()
    assert sorted(removed) == sorted(first)
    assert not [p for p in (tmp_path / "xdg").rglob("*") if p.is_file()]
    assert desktop.uninstall() == []  # nothing left, no error


@linux_only
def test_uninstall_keeps_foreign_files(tmp_path):
    other = tmp_path / "xdg" / "applications" / "other.desktop"
    other.parent.mkdir(parents=True)
    other.write_text("keep")
    desktop.install(command=["/x/tradebot"])
    desktop.uninstall()
    assert other.read_text() == "keep"


@linux_only
def test_exec_quotes_paths_with_spaces_and_special_characters(tmp_path):
    desktop.install(command=["/home/me/my apps/trade bot/tradebot"])
    assert exec_of(tmp_path / "xdg" / "applications" / "tradebot.desktop") == '"/home/me/my apps/trade bot/tradebot" gui'
    assert desktop.quote_arg('/a b/$c"d`e\\f%g') == '"/a b/\\\\$c\\\\"d\\\\`e\\\\\\\\f%%g"'
    assert desktop.quote_arg("/plain/path") == "/plain/path"


@linux_only
def test_dev_entry_is_separate(tmp_path):
    desktop.install(command=["/x/tradebot"])
    desktop.install(dev=True, command=["/src/.venv/bin/tradebot"])
    apps = tmp_path / "xdg" / "applications"
    dev = (apps / "tradebot-dev.desktop").read_text()
    assert "Name=Tradebot (dev)" in dev and "Icon=tradebot-dev" in dev
    assert exec_of(apps / "tradebot-dev.desktop") == "env TRADEBOT_DESKTOP_ID=tradebot-dev /src/.venv/bin/tradebot gui"
    assert not (tmp_path / "xdg" / "metainfo" / "x").exists()
    assert not list((tmp_path / "xdg" / "metainfo").glob("*dev*"))
    desktop.uninstall(dev=True)
    assert (apps / "tradebot.desktop").is_file() and not (apps / "tradebot-dev.desktop").exists()
    assert (tmp_path / "xdg" / "icons" / "hicolor" / "48x48" / "apps" / "tradebot.png").is_file()


@linux_only
def test_xdg_data_home_and_prefix(tmp_path, monkeypatch):
    assert desktop.data_dir() == tmp_path / "xdg"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert desktop.data_dir() == tmp_path / "home" / ".local" / "share"
    assert desktop.data_dir(system=True) == desktop.Path("/usr/local/share")
    desktop.install(prefix=tmp_path / "pre", command=["/x/tradebot"])
    assert (tmp_path / "pre" / "share" / "applications" / "tradebot.desktop").is_file()


@linux_only
def test_dry_run_writes_nothing(tmp_path):
    files = desktop.install(dry_run=True, command=["/x/tradebot"])
    assert files and not (tmp_path / "xdg").exists()
    desktop.install(command=["/x/tradebot"])
    assert desktop.uninstall(dry_run=True)
    assert (tmp_path / "xdg" / "applications" / "tradebot.desktop").exists()


@linux_only
def test_system_install_needs_root(monkeypatch):
    monkeypatch.setattr(desktop.os, "geteuid", lambda: 1000)
    with pytest.raises(RuntimeError, match="root"):
        desktop.install(system=True, command=["/x/tradebot"])


def test_refused_on_unsupported_systems(monkeypatch):
    monkeypatch.setattr(desktop.sys, "platform", "freebsd14")
    with pytest.raises(RuntimeError, match="not supported"):
        desktop.install()
    result = runner.invoke(app, ["desktop", "install"])
    assert result.exit_code == 1 and "not supported" in result.output


# ---- Windows and macOS run on fakes: the platform is patched, shortcut creation is recorded, nothing leaves tmp_path

@pytest.fixture
def windows(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "ProgramData"))
    made = []

    def fake(lnk, target, arguments, icon):
        lnk.write_text("fake")
        made.append((lnk.name, target, arguments, icon))

    monkeypatch.setattr(desktop, "write_shortcut", fake)
    return made


def test_windows_start_menu_shortcut(tmp_path, windows, monkeypatch):
    monkeypatch.setattr(desktop.sys, "frozen", True, raising=False)
    exe = tmp_path / "tb" / "tradebot.exe"
    files = desktop.install(command=[str(exe)])
    menu = tmp_path / "AppData" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    assert files == [menu / "Tradebot.lnk"] and files[0].is_file()
    name, target, args, icon = windows[0]
    assert target == str(exe.with_name("tradebot-gui.exe")) and args == "" and icon.endswith("tradebot.ico")
    assert desktop.install(command=[str(exe)]) == files  # idempotent
    assert desktop.uninstall() == files and not files[0].exists()
    assert desktop.uninstall() == []


def test_windows_dev_shortcut_uses_pythonw_and_system_uses_programdata(tmp_path, windows):
    venv = tmp_path / "repo" / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "pythonw.exe").write_text("")
    (venv / "python.exe").write_text("")
    [lnk] = desktop.install(dev=True, command=[str(venv / "tradebot.exe")])
    assert lnk.name == "Tradebot (dev).lnk"
    assert windows[0][1:3] == (str(venv / "pythonw.exe"), "-m tradebot.cli.main gui")
    assert desktop.start_menu_dir(system=True) == tmp_path / "ProgramData" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    [lnk2] = desktop.install(prefix=tmp_path / "pre", command=[str(venv / "tradebot.exe")])
    assert lnk2 == tmp_path / "pre" / "Tradebot.lnk"


@pytest.fixture
def macos(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop.sys, "platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def fake_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "dl" / "tradebot.app"
    (bundle / "Contents" / "MacOS").mkdir(parents=True)
    exe = bundle / "Contents" / "MacOS" / "tradebot"
    exe.write_text("bin")
    (bundle / "Contents" / "Info.plist").write_text("plist")
    (bundle / "Contents" / "link").symlink_to("Info.plist")
    monkeypatch.setattr(desktop.sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop.sys, "executable", str(exe))
    return bundle


def test_macos_copies_the_running_bundle_into_applications(macos, monkeypatch):
    bundle = fake_bundle(macos, monkeypatch)
    [app] = desktop.install()
    assert app == macos / "home" / "Applications" / "Tradebot.app"
    assert (app / "Contents" / "MacOS" / "tradebot").read_text() == "bin"
    assert (app / "Contents" / "link").is_symlink()
    assert desktop.install() == [app]  # idempotent
    assert desktop.uninstall() == [app] and not app.exists() and bundle.exists()
    assert desktop.uninstall() == []


def test_macos_system_and_prefix_locations(macos):
    assert desktop.applications_dir(system=True) == desktop.Path("/Applications")
    assert desktop.applications_dir(prefix=macos / "p") == macos / "p" / "Applications"


def test_macos_dev_stub_app(macos):
    [app] = desktop.install(dev=True, command=["/src/my repo/.venv/bin/tradebot"])
    assert app.name == "Tradebot (dev).app"
    launcher = app / "Contents" / "MacOS" / "launch"
    assert launcher.read_text() == "#!/bin/sh\nexec '/src/my repo/.venv/bin/tradebot' gui\n"
    assert launcher.stat().st_mode & 0o111
    info = desktop.plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleName"] == "Tradebot (dev)" and info["LSApplicationCategoryType"] == "public.app-category.finance"
    assert (app / "Contents" / "Resources" / "tradebot.icns").is_file()
    assert desktop.uninstall(dev=True) == [app] and not app.exists()


def test_macos_pip_install_gets_a_launcher_stub_named_tradebot(macos):
    [app] = desktop.install(command=["/venv/bin/tradebot"])
    assert app.name == "Tradebot.app" and (app / "Contents" / "MacOS" / "launch").is_file()


def test_icons_are_valid_containers():
    import struct

    ico = (desktop.ASSETS / "tradebot.ico").read_bytes()
    assert struct.unpack("<HHH", ico[:6]) == (0, 1, 5)
    icns = (desktop.ASSETS / "tradebot.icns").read_bytes()
    assert icns[:4] == b"icns" and struct.unpack(">I", icns[4:8])[0] == len(icns)
    assert (desktop.ASSETS / "tradebot.svg").read_text().startswith("<?xml")


@linux_only
def test_cli_install_and_uninstall(tmp_path):
    prefix = str(tmp_path / "p")
    result = runner.invoke(app, ["desktop", "install", "--prefix", prefix])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "p" / "share" / "applications" / "tradebot.desktop").is_file()
    result = runner.invoke(app, ["desktop", "uninstall", "--prefix", prefix])
    assert result.exit_code == 0
    assert not (tmp_path / "p" / "share" / "applications" / "tradebot.desktop").exists()


def test_launcher_command_for_frozen_and_script(monkeypatch, tmp_path):
    exe = tmp_path / "tradebot"
    exe.write_text("")
    monkeypatch.setattr(desktop.sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop.sys, "executable", str(exe))
    assert desktop.launcher_command() == [str(exe)]
    monkeypatch.delattr(desktop.sys, "frozen")
    monkeypatch.setattr(desktop.sys, "argv", [str(exe)])
    assert desktop.launcher_command() == [str(exe)]


@linux_only
@pytest.mark.skipif(shutil.which("desktop-file-validate") is None, reason="desktop-file-utils not installed")
def test_desktop_files_validate(tmp_path):
    desktop.install(command=["/opt/my apps/tradebot"])
    desktop.install(dev=True, command=["/opt/my apps/tradebot"])
    for name in ("tradebot", "tradebot-dev"):
        done = subprocess.run(["desktop-file-validate", str(tmp_path / "xdg" / "applications" / f"{name}.desktop")],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(shutil.which("appstreamcli") is None, reason="appstream not installed")
def test_metainfo_validates():
    done = subprocess.run(["appstreamcli", "validate", "--no-net", str(desktop.ASSETS / f"{desktop.APP_ID}.metainfo.xml")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
