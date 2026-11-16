"""The worked example of docs/indicators-as-algorithms.md loads as a plugin and behaves like a built-in."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.algorithms.pool import default_pool
from tradebot.backtest import backtest
from tradebot.core import ops
from tradebot.core.registry import load_plugins
from tradebot.core.series import OHLC
from tradebot.indicators import INDICATORS
from .conftest import make_ohlc

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "plugins" / "zscore_reversion.py"


@pytest.fixture
def plugin(tmp_path):
    shutil.copy(EXAMPLE, tmp_path / "zscore_reversion.py")
    loaded = load_plugins([tmp_path])
    assert loaded == ["tradebot_plugin_zscore_reversion"]
    yield
    ALGORITHMS.unregister("zscore_reversion")
    INDICATORS.unregister("zscore")


def test_example_registers_and_backtests(plugin):
    assert "zscore_reversion" in ALGORITHMS and "zscore" in INDICATORS
    df = make_ohlc(600, seed=3)
    algo = build_algorithm("zscore_reversion", {"entry": 1.5})
    pos = algo.run(df)
    assert set(np.unique(pos)) <= {0.0, 1.0} and pos.sum() > 0
    assert backtest(df, algo).metrics["trades"] > 0
    assert [i.name for i in algo.chart_indicators()] == ["zscore"]


def test_example_is_vectorized_and_causal(plugin):
    algo = build_algorithm("zscore_reversion")
    assert algo.vectorized
    df = make_ohlc(400, seed=5)
    # a batch of float parameters ([population, 1]) gives one row of positions per candidate
    p = {"period": 20, "entry": np.array([[1.0], [2.0], [3.0]]), "exit": np.array([[0.0], [0.0], [0.0]])}
    assert algo.signals(OHLC.from_frame(df), p).shape == (3, len(df))
    # no look-ahead: positions on the first 300 candles are unchanged when later candles are added
    assert np.array_equal(algo.run(df.iloc[:300]).to_numpy(), algo.run(df).to_numpy()[:300])


def test_example_joins_the_pool_and_vote(plugin):
    members, _ = default_pool()
    assert "zscore_reversion" in members
    vote = build_algorithm("vote", members=["zscore_reversion", "sma_cross"])
    assert "zscore_reversion.entry" in vote.param_space()
    assert vote.run(make_ohlc(300)).shape == (300,)


def test_alert_output(plugin):
    o = OHLC.from_frame(make_ohlc(300, seed=1))
    out = INDICATORS.get("zscore").compute(o, period=20)
    assert np.isnan(out["oversold"][:19]).all()
    assert set(np.unique(out["oversold"][~np.isnan(out["oversold"])])) <= {0.0, 1.0}
    assert ops.isnan(out["value"][:19]).all()
