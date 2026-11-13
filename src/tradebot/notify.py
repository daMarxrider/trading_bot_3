# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Notifications (#155): push bot trades, errors, risk-guard stops and scanner hits to your phone or desktop.

Channels live in ``config.toml``, one table each; the table name is free, ``type`` picks the channel::

    [notify.phone]
    type = "telegram"
    token = "$TELEGRAM_TOKEN"       # "$NAME" / "env:NAME": read from the environment, so the secret stays out of the file
    chat_id = "12345678"
    events = ["trade", "error"]     # optional filter; default: all kinds

    [notify.team]
    type = "discord"
    url = "env:DISCORD_WEBHOOK"

    [notify.hook]                   # generic: POSTs {"kind", "title", "text"} as JSON
    type = "webhook"
    url = "https://example.com/hook"

    [notify.mail]
    type = "email"
    host = "smtp.example.com"
    port = 587                      # 465: SSL from the start, otherwise STARTTLS when offered
    user = "me@example.com"
    password = "$SMTP_PASSWORD"
    to = "me@example.com"           # optional "from" (default: user)

    [notify.pc]
    type = "desktop"                # notify-send (Linux), osascript (macOS) or the app's tray icon

Event kinds (:data:`KINDS`): ``trade`` (a bot bought/sold), ``error`` (a bot step failed), ``risk`` (a risk guard
fired, the hook for #154: ``notify("risk", title, text)``), ``scanner`` and ``test``. ``enabled = false`` mutes a channel.

:func:`notify` never blocks and never raises: each send runs in a daemon thread with a timeout, and a failure is
only logged (without secrets). Add a channel type with :func:`channel`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import smtplib
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import EmailMessage

import requests

from tradebot.config import load_config

log = logging.getLogger(__name__)

KINDS = ("trade", "error", "risk", "scanner", "test")
TIMEOUT = 10.0  #: seconds for every network call

#: Channel types: name -> (send function(options, kind, title, text), the options a channel needs)
CHANNELS: dict[str, tuple[Callable[[dict, str, str, str], None], tuple[str, ...]]] = {}

#: The options each channel type reads, in the order the settings dialog shows them
CHANNEL_OPTIONS = {"telegram": ("token", "chat_id"), "discord": ("url",), "webhook": ("url",),
                   "email": ("host", "port", "user", "password", "to", "from"), "desktop": ()}

#: The GUI sets this to show a tray message when no desktop notifier is installed: ``fallback(title, text)``
desktop_fallback: Callable[[str, str], None] | None = None


def channel(name: str, required: tuple[str, ...] = ()):
    """Register a channel type: ``@channel("sms", ("url",)) def _sms(options, kind, title, text)``."""
    def register(fn):
        CHANNELS[name] = (fn, required)
        return fn
    return register


def resolve(value):
    """``"$NAME"`` / ``"env:NAME"`` -> the environment variable's value (empty if unset); anything else as is."""
    if isinstance(value, str):
        for prefix in ("$", "env:"):
            if value.startswith(prefix) and len(value) > len(prefix):
                return os.environ.get(value[len(prefix):], "")
    return value


@dataclass
class Channel:
    name: str
    type: str
    options: dict = field(repr=False)  # holds secrets: never in a repr or a log
    events: tuple[str, ...] = KINDS
    enabled: bool = True

    def wants(self, kind: str) -> bool:
        return self.enabled and (kind == "test" or kind in self.events)

    def missing(self) -> list[str]:
        return [k for k in CHANNELS[self.type][1] if not resolve(self.options.get(k))]


def channels(config: dict | None = None) -> list[Channel]:
    """The configured channels (``[notify.*]``); a malformed table is skipped with a warning."""
    table = (load_config() if config is None else config).get("notify", {})
    out = []
    for name, opts in (table.items() if isinstance(table, dict) else ()):
        if not isinstance(opts, dict) or opts.get("type") not in CHANNELS:
            log.warning("notify.%s: unknown or missing type (one of: %s)", name, ", ".join(CHANNELS))
            continue
        events = opts.get("events", KINDS)
        events = (events,) if isinstance(events, str) else tuple(events)
        rest = {k: v for k, v in opts.items() if k not in ("type", "events", "enabled")}
        out.append(Channel(str(name), opts["type"], rest, events, bool(opts.get("enabled", True))))
    return out


def send(ch: Channel, kind: str, title: str, text: str) -> str | None:
    """Send to one channel and wait. Returns None on success, else a short error (never containing secrets)."""
    if ch.missing():
        return f"missing setting: {', '.join(ch.missing())}"
    try:
        CHANNELS[ch.type][0](ch.options, kind, title, text)
    except Exception as exc:  # a notification must never take the bot down
        msg = _scrub(f"{type(exc).__name__}: {exc}", ch)
        log.warning("notification via %s (%s) failed: %s", ch.name, ch.type, msg)
        return msg
    return None


def _scrub(msg: str, ch: Channel) -> str:
    """Remove the channel's secret values (a requests error contains the URL, so the bot token) from an error text."""
    for v in ch.options.values():
        v = str(resolve(v))
        if len(v) >= 4:
            msg = msg.replace(v, "***")
    return msg


def notify(kind: str, title: str, text: str = "", wait: bool = False) -> int:
    """Tell every channel that listens to ``kind``. Returns how many channels were started.

    Non-blocking (a daemon thread per channel) unless ``wait``; never raises, also with a broken config.
    """
    try:
        targets = [c for c in channels() if c.wants(kind)]
        threads = [threading.Thread(target=send, args=(c, kind, title, text), daemon=True, name=f"notify-{c.name}")
                   for c in targets]
        for t in threads:
            t.start()
        if wait:
            for t in threads:
                t.join()
        return len(threads)
    except Exception:
        log.exception("could not send notifications")
        return 0


def test_all() -> dict[str, str | None]:
    """Send a test message to every enabled channel (whatever its event filter), waiting: name -> error or None."""
    return {c.name: send(c, "test", "tradebot test", "Notifications work.") for c in channels() if c.enabled}


test_all.__test__ = False  # not a pytest test


# -- channel types ---------------------------------------------------------------------------------
@channel("telegram", ("token", "chat_id"))
def _telegram(o, kind, title, text):
    r = requests.post(f"https://api.telegram.org/bot{resolve(o['token'])}/sendMessage",
                      json={"chat_id": str(resolve(o["chat_id"])), "text": f"{title}\n{text}".strip()}, timeout=TIMEOUT)
    r.raise_for_status()


@channel("discord", ("url",))
def _discord(o, kind, title, text):
    requests.post(resolve(o["url"]), json={"content": f"**{title}**\n{text}"[:2000]}, timeout=TIMEOUT).raise_for_status()


@channel("webhook", ("url",))
def _webhook(o, kind, title, text):
    requests.post(resolve(o["url"]), json={"kind": kind, "title": title, "text": text}, timeout=TIMEOUT).raise_for_status()


@channel("email", ("host", "to"))
def _email(o, kind, title, text):
    user, password = resolve(o.get("user", "")), resolve(o.get("password", ""))
    msg = EmailMessage()
    msg["Subject"], msg["To"] = title, resolve(o["to"])
    msg["From"] = resolve(o.get("from", "")) or user or resolve(o["to"])
    msg.set_content(text or title)
    host, port = resolve(o["host"]), int(o.get("port", 587))
    server = smtplib.SMTP_SSL(host, port, timeout=TIMEOUT) if port == 465 else smtplib.SMTP(host, port, timeout=TIMEOUT)
    with server:
        if port != 465 and server.has_extn("starttls"):
            server.starttls()
        if user:
            server.login(user, password)
        server.send_message(msg)


@channel("desktop")
def _desktop(o, kind, title, text):
    if sys.platform == "darwin" and shutil.which("osascript"):
        script = f"display notification {json.dumps(text)} with title {json.dumps(title)}"
        subprocess.run(["osascript", "-e", script], check=True, timeout=TIMEOUT, capture_output=True)
    elif shutil.which("notify-send"):
        subprocess.run(["notify-send", title, text], check=True, timeout=TIMEOUT, capture_output=True)
    elif desktop_fallback:
        desktop_fallback(title, text)
    else:
        raise RuntimeError("no desktop notifier found (install notify-send, or run the app for tray messages)")


# -- sources ---------------------------------------------------------------------------------------
def bot_event(event) -> None:
    """Notify about a :class:`tradebot.live.BotEvent`; holds are silent."""
    if event.action == "hold":
        return
    mode = " (dry run)" if event.dry_run else ""
    notify("trade", f"{event.action.upper()} {event.symbol}{mode}",
           f"{event.amount:.6g} at {event.price:.6g}" + (f"\n{event.note}" if event.note else ""))
