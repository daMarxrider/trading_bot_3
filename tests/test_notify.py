# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Notification channels (#155): every network call is mocked."""

import os
import smtplib
import subprocess
import threading
from types import SimpleNamespace

import pytest
import requests
from typer.testing import CliRunner

from tradebot import notify
from tradebot.config import load_config

CONFIG = """
[notify.phone]
type = "telegram"
token = "$TG_TOKEN"
chat_id = "42"
events = ["trade"]

[notify.team]
type = "discord"
url = "https://discord.test/hook/SECRET123"

[notify.hook]
type = "webhook"
url = "https://example.test/hook"
events = ["risk", "error"]

[notify.off]
type = "webhook"
url = "https://example.test/off"
enabled = false
"""


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    monkeypatch.setenv("TRADEBOT_CONFIG", str(path))
    monkeypatch.setenv("TG_TOKEN", "123:SECRETTOKEN")
    return path


@pytest.fixture
def posts(monkeypatch):
    calls = []

    def post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(requests, "post", post)
    return calls


def test_event_filters_and_env_secrets(config, posts):
    assert notify.notify("trade", "BUY X", "1 at 2", wait=True) == 2  # phone (trade) + team (all kinds)
    urls = sorted(u for u, _, _ in posts)
    assert urls == ["https://api.telegram.org/bot123:SECRETTOKEN/sendMessage", "https://discord.test/hook/SECRET123"]
    assert all(t == notify.TIMEOUT for _, _, t in posts)
    posts.clear()
    assert notify.notify("risk", "Drawdown stop", "x", wait=True) == 2  # team + hook; "off" is muted
    hook = next(j for u, j, _ in posts if u == "https://example.test/hook")
    assert hook == {"kind": "risk", "title": "Drawdown stop", "text": "x"}


def test_no_config_does_nothing(tmp_path, monkeypatch, posts):
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "none.toml"))
    assert notify.notify("trade", "t", wait=True) == 0 and not posts


def test_failure_never_raises_and_hides_secrets(config, monkeypatch, caplog):
    def boom(url, **kw):
        raise requests.ConnectionError(f"cannot reach {url}")

    monkeypatch.setattr(requests, "post", boom)
    notify.notify("trade", "t", "x", wait=True)  # no exception
    phone = next(c for c in notify.channels() if c.name == "phone")
    error = notify.send(phone, "trade", "t", "x")
    assert "ConnectionError" in error and "SECRETTOKEN" not in error and "SECRETTOKEN" not in caplog.text


def test_broken_config_does_not_raise(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("[notify.x\n")
    monkeypatch.setenv("TRADEBOT_CONFIG", str(path))
    assert notify.notify("trade", "t", wait=True) == 0


def test_missing_setting_is_reported(tmp_path, monkeypatch, posts):
    (tmp_path / "c.toml").write_text('[notify.t]\ntype = "telegram"\ntoken = "$NOPE_UNSET"\nchat_id = "1"\n')
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "c.toml"))
    monkeypatch.delenv("NOPE_UNSET", raising=False)
    assert notify.test_all() == {"t": "missing setting: token"} and not posts


def test_test_all_ignores_the_event_filter(config, posts):
    assert notify.test_all() == {"phone": None, "team": None, "hook": None}  # "off" is disabled
    assert len(posts) == 3


def test_email(tmp_path, monkeypatch):
    (tmp_path / "c.toml").write_text('[notify.m]\ntype = "email"\nhost = "smtp.test"\nport = 587\nuser = "me@x"\n'
                                     'password = "$SMTP_PW"\nto = "you@x"\n')
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "c.toml"))
    monkeypatch.setenv("SMTP_PW", "pw")
    log = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            log.append(("connect", host, port, timeout))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def has_extn(self, name):
            return True

        def starttls(self):
            log.append("starttls")

        def login(self, user, password):
            log.append(("login", user, password))

        def send_message(self, msg):
            log.append(("send", msg["Subject"], msg["To"], msg["From"]))

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    assert notify.test_all() == {"m": None}
    assert log == [("connect", "smtp.test", 587, notify.TIMEOUT), "starttls", ("login", "me@x", "pw"),
                   ("send", "tradebot test", "you@x", "me@x")]


def test_desktop_uses_notify_send_then_the_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(notify.sys, "platform", "linux")
    monkeypatch.setattr(notify.shutil, "which", lambda name: "/usr/bin/notify-send")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    ch = notify.Channel("pc", "desktop", {})
    assert notify.send(ch, "trade", "T", "body") is None and calls == [["notify-send", "T", "body"]]
    monkeypatch.setattr(notify.shutil, "which", lambda name: None)
    assert "no desktop notifier" in notify.send(ch, "trade", "T", "body")
    shown = []
    monkeypatch.setattr(notify, "desktop_fallback", lambda t, b: shown.append((t, b)))
    assert notify.send(ch, "trade", "T", "body") is None and shown == [("T", "body")]


def test_bot_events_are_notified_holds_are_not(config, posts):
    def event(action):
        return SimpleNamespace(action=action, symbol="BTC/USDT", dry_run=True, amount=1.5, price=100.0, note="")

    notify.bot_event(event("hold"))
    assert not posts
    notify.bot_event(event("buy"))
    for t in threading.enumerate():
        if t.name.startswith("notify-"):
            t.join(5)
    assert any("BUY BTC/USDT" in str(j) for _, j, _ in posts)


def test_cli_notify_test(config, posts):
    from tradebot.cli.main import app

    result = CliRunner().invoke(app, ["notify", "test"])
    assert result.exit_code == 0 and "phone" in result.output and len(posts) == 3


def test_cli_notify_test_without_channels(tmp_path, monkeypatch):
    from tradebot.cli.main import app

    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "none.toml"))
    assert CliRunner().invoke(app, ["notify", "test"]).exit_code == 1


def test_settings_page_saves_and_tests(tmp_path, monkeypatch, posts):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from tradebot.gui.settings import SettingsDialog
    from tradebot.gui.theme import ThemeManager

    app = QApplication.instance() or QApplication([])  # noqa: F841
    monkeypatch.setenv("TRADEBOT_CONFIG", str(tmp_path / "config.toml"))
    themes = ThemeManager(user_dir=tmp_path / "themes")
    themes.apply("dark")
    dialog = SettingsDialog(themes, "dark")
    try:
        for key, text in (("token", "tok"), ("chat_id", "7")):
            dialog.notify_fields["telegram", key].setText(text)
            dialog.notify_fields["telegram", key].editingFinished.emit()
        dialog.notify_events["telegram"]["scanner"].setChecked(False)
        conf = load_config()["notify"]["telegram"]
        assert conf["type"] == "telegram" and conf["token"] == "tok" and conf["chat_id"] == "7"
        assert "scanner" not in conf["events"] and conf["enabled"] is True
        assert dialog.test_notifications() == {"telegram": None}
        assert posts[0][0].endswith("bottok/sendMessage")
        assert "✓ telegram" in dialog.notify_result.text()
    finally:
        dialog.close()
        dialog.deleteLater()
