# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Background alerts (#156): price/indicator alerts, scheduled scans, the CLI. No network, no threads."""

import pandas as pd
import pytest
from typer.testing import CliRunner

from tradebot import alerts

from .conftest import make_ohlc


@pytest.fixture
def path(tmp_path, monkeypatch):
    p = tmp_path / "alerts.json"
    monkeypatch.setenv("TRADEBOT_ALERTS", str(p))
    return p


class Feed:
    """Fake candles: a price per symbol, constant history."""

    def __init__(self, **prices):
        self.prices = prices

    def __call__(self, symbol, timeframe):
        df = make_ohlc(60)
        df["close"] = float(self.prices.get(symbol, 100.0))
        return df


class Sink:
    def __init__(self):
        self.sent = []

    def __call__(self, kind, title, text):
        self.sent.append((kind, title, text))


def test_price_alert_fires_once_and_disables(path):
    alerts.add(alerts.Alert("price", ["BTC/USDT"], op=">=", value=100))
    sink, feed = Sink(), Feed(**{"BTC/USDT": 90})
    assert alerts.run_due(feed, sink, now=1000, path=path) == []
    feed.prices["BTC/USDT"] = 101
    lines = alerts.run_due(feed, sink, now=1100, path=path)
    assert len(lines) == 1 and sink.sent[0][0] == "alert" and "101" in sink.sent[0][2]
    assert alerts.load(path)[0].enabled is False
    alerts.run_due(feed, sink, now=2000, path=path)
    assert len(sink.sent) == 1


def test_repeat_alert_rearms_after_condition_clears(path):
    alerts.add(alerts.Alert("price", ["X"], op="<=", value=50, repeat=True))
    sink, feed = Sink(), Feed(X=40)
    alerts.run_due(feed, sink, now=1000, path=path)
    alerts.run_due(feed, sink, now=1100, path=path)  # still below: no second message
    assert len(sink.sent) == 1
    feed.prices["X"] = 60
    alerts.run_due(feed, sink, now=1200, path=path)
    feed.prices["X"] = 45
    alerts.run_due(feed, sink, now=1300, path=path)
    assert len(sink.sent) == 2


def test_not_due_before_interval(path):
    alerts.add(alerts.Alert("price", ["X"], op=">=", value=1, every=5))
    sink = Sink()
    alerts.run_due(Feed(X=0.5), sink, now=1000, path=path)
    called = []
    alerts.run_due(lambda *a: called.append(a), sink, now=1100, path=path)  # 100 s < 5 min
    assert not called


def test_every_close_waits_for_next_candle(path):
    a = alerts.Alert("price", ["X"], "1h", every="close")
    a.last_run = 3600 * 10 + 5
    assert not alerts.is_due(a, 3600 * 10 + 3000)
    assert alerts.is_due(a, 3600 * 11)


def test_indicator_alert(path):
    alerts.add(alerts.Alert("indicator", ["AAA"], "1d", indicator="rsi",
                            filters=[{"output": "value", "op": "<", "value": 30}], every="close"))
    sink = Sink()
    flat_down = lambda s, tf: pd.DataFrame(  # noqa: E731
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [100 - i for i in range(60)], "volume": 1.0},
        index=pd.date_range("2020-01-01", periods=60, tz="UTC"))
    lines = alerts.run_due(flat_down, sink, now=10**6, path=path)
    assert lines and sink.sent[0][1] == "Indicator alert AAA"


def test_scan_reports_only_new_matches(path):
    alerts.add(alerts.Alert("scan", ["AAA", "BBB"], "1d", indicator="rsi",
                            filters=[{"output": "value", "op": ">", "value": 0}], every=1))
    sink = Sink()
    alerts.run_due(Feed(), sink, now=1000, path=path)
    assert sink.sent[0][0] == "scanner" and "AAA" in sink.sent[0][2] and "BBB" in sink.sent[0][2]
    alerts.run_due(Feed(), sink, now=1100, path=path)  # same matches: silent
    assert len(sink.sent) == 1
    assert set(alerts.load(path)[0].last_matches) == {"AAA", "BBB"}


def test_failing_alert_does_not_stop_others(path):
    alerts.add(alerts.Alert("price", ["BAD"], op=">=", value=1))
    alerts.add(alerts.Alert("price", ["OK"], op=">=", value=1))
    sink = Sink()

    def feed(sym, tf):
        if sym == "BAD":
            raise RuntimeError("no network")
        return Feed(OK=5)(sym, tf)

    lines = alerts.run_due(feed, sink, now=1000, path=path)
    assert any("failed" in line for line in lines) and len(sink.sent) == 1


def test_validate_rejects_bad_alerts():
    for bad in (alerts.Alert("price", [], op=">="), alerts.Alert("price", ["X"], op="=="),
                alerts.Alert("price", ["X"], "2s"), alerts.Alert("price", ["X"], every="abc"),
                alerts.Alert("indicator", ["X"], indicator="nope"),
                alerts.Alert("indicator", ["X"], indicator="rsi"),
                alerts.Alert("indicator", ["X"], indicator="rsi", filters=[{"output": "zzz", "op": "<", "value": 1}])):
        with pytest.raises(ValueError):
            alerts.validate(bad)


def test_store_roundtrip_remove_and_corrupt(path):
    a = alerts.add(alerts.Alert("price", ["X"], op=">", value=3))
    assert alerts.load(path)[0].id == a.id
    assert alerts.remove(a.id) and not alerts.remove(a.id)
    path.write_text("not json")
    assert alerts.load(path) == []


def test_watch_stops(path):
    alerts.add(alerts.Alert("price", ["X"], op=">=", value=1))
    out, calls = [], iter([False, True])
    alerts.watch(0, stop=lambda: next(calls), log_fn=out.append, fetch=Feed(X=2), notify_fn=Sink())
    assert len(out) == 1


def test_parse_price():
    assert alerts.parse_price(" >=100000") == (">=", 100000.0)
    with pytest.raises(ValueError):
        alerts.parse_price("100")


def test_cli(path, monkeypatch):
    from tradebot.cli.main import app

    run = CliRunner().invoke
    r = run(app, ["alert", "price", "BTC/USDT", ">=100000", "--every", "5"])
    assert r.exit_code == 0 and "added" in r.output
    r = run(app, ["alert", "indicator", "AAPL", "-i", "rsi", "--where", "value < 30"])
    assert r.exit_code == 0, r.output
    r = run(app, ["alert", "price", "X", "100"])
    assert r.exit_code != 0
    r = run(app, ["alert", "price", "X", ">1", "--every", "abc"])
    assert r.exit_code != 0
    r = run(app, ["alert", "list"])
    assert "BTC/USDT" in r.output and "value < 30" in r.output
    first = alerts.load(path)[0].id
    assert run(app, ["alert", "rm", first]).exit_code == 0
    assert run(app, ["alert", "rm", first]).exit_code != 0
    monkeypatch.setattr(alerts, "run_due", lambda: ["a1 fired"])
    r = run(app, ["watch", "--once"])
    assert r.exit_code == 0 and "a1 fired" in r.output
