# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Strategy parameters as the user types them (-p key=value, the app's parameter field) and the registries."""

import pytest

from tradebot.algorithms import ALGORITHMS, build_algorithm
from tradebot.core.registry import IntParam, Param, Registry, load_plugins, resolve_params

SPACE = {"period": IntParam(14, 2, 50), "level": Param(30.0, 5.0, 45.0)}
FLAG = {"on": IntParam(1, 0, 1)}


def integer_params(flags: bool) -> list[tuple[str, str]]:
    """``(strategy, parameter)`` of every registered strategy's integer parameters: the on/off flags (range from
    0) or the window lengths and counts (range from 1 or more)."""
    out = []
    for name in ALGORITHMS.names():
        for key, spec in build_algorithm(name, {}).param_space().items():
            if spec.integer and (spec.low < 1) == flags:
                out.append((name, key))
    return out


def test_missing_parameters_take_their_defaults():
    assert resolve_params(SPACE, {}) == {"period": 14, "level": 30.0}


def test_integer_parameters_come_out_as_int():
    out = resolve_params(SPACE, {"period": 20.0})
    assert out["period"] == 20 and isinstance(out["period"], int)


def test_numbers_written_as_text_are_read():
    assert resolve_params(SPACE, {"period": "21", "level": "25.5"}) == {"period": 21, "level": 25.5}


def test_an_unknown_parameter_is_rejected_listing_the_known_ones():
    with pytest.raises(ValueError, match=r"unknown parameter\(s\) \['lvl'\]; expected \['level', 'period'\]"):
        resolve_params(SPACE, {"lvl": 25})


@pytest.mark.parametrize("value", ["abc", "", None, [1, 2]])
def test_a_value_that_is_not_a_number_is_rejected_by_name(value):
    with pytest.raises(ValueError, match="parameter level"):
        resolve_params(SPACE, {"level": value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "nan"])
@pytest.mark.parametrize("name", ["period", "level"])
def test_a_value_that_is_not_finite_is_rejected_by_name(name, value):
    with pytest.raises(ValueError, match=f"parameter {name}.*finite"):
        resolve_params(SPACE, {name: value})


def test_a_real_strategy_refuses_a_nan_parameter():
    with pytest.raises(ValueError, match="finite"):
        build_algorithm("rsi_threshold", {"lower": float("nan")})


def test_with_params_validates_like_the_constructor():
    algo = build_algorithm("rsi_threshold", {})
    with pytest.raises(ValueError, match="unknown parameter"):
        algo.with_params(nope=1)


@pytest.mark.parametrize("value, expected", [(1.0, 2), (99.0, 50), (14.4, 14), (14.6, 15)])
def test_clip_keeps_integers_inside_the_range_and_rounds(value, expected):
    assert SPACE["period"].clip(value) == expected


def test_a_positional_description_never_makes_a_float_param_an_integer():
    assert Param(1.0, 0.0, 2.0, "a description").integer is False


def test_an_unknown_name_lists_what_is_available():
    reg = Registry("widget")

    class Thing:
        name = "thing"

    reg.register(Thing)
    with pytest.raises(KeyError, match="unknown widget 'thng'; available: thing"):
        reg.get("thng")


def test_a_class_without_a_name_is_not_registered():
    reg = Registry("widget")

    class Anonymous:
        name = ""

    reg.register(Anonymous)
    assert len(reg) == 0


def test_unregistering_an_unknown_name_is_ignored():
    reg = Registry("widget")
    reg.unregister("never-there")
    assert len(reg) == 0


def test_an_unknown_strategy_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown algorithm 'rsi_treshold'"):
        ALGORITHMS.get("rsi_treshold")


def test_a_broken_plugin_is_skipped_and_the_good_one_still_loads(tmp_path, caplog):
    (tmp_path / "broken.py").write_text("raise RuntimeError('plugin bug')\n")
    (tmp_path / "good.py").write_text("VALUE = 1\n")
    (tmp_path / "_private.py").write_text("raise RuntimeError('never imported')\n")
    loaded = load_plugins([tmp_path])
    assert loaded == ["tradebot_plugin_good"]
    assert "failed to load plugin" in caplog.text


def test_a_missing_plugin_folder_loads_nothing(tmp_path):
    assert load_plugins([tmp_path / "missing"]) == []


# -- integer parameters: windows can't be 0 or negative, flags can be 0 ------------------------------------------
@pytest.mark.parametrize("value", [-1, -14, -1.5])
def test_a_negative_integer_parameter_is_rejected(value):
    with pytest.raises(ValueError, match="parameter period.*too small, it must be at least 1"):
        resolve_params(SPACE, {"period": value})


def test_a_window_of_zero_is_rejected():
    with pytest.raises(ValueError, match="parameter period: 0 is too small"):
        resolve_params(SPACE, {"period": 0})


def test_a_window_below_the_search_range_but_at_least_one_is_allowed():
    assert resolve_params(SPACE, {"period": 1}) == {"period": 1, "level": 30.0}  # the range is only the search's


def test_a_flag_can_be_switched_off():
    assert resolve_params(FLAG, {"on": 0}) == {"on": 0}


def test_a_flag_cannot_be_negative():
    with pytest.raises(ValueError, match="parameter on: -1 is too small, it must be at least 0"):
        resolve_params(FLAG, {"on": -1})


def test_negative_float_parameters_are_allowed():
    assert resolve_params({"x": Param(0.0, -1.0, 1.0)}, {"x": -0.5}) == {"x": -0.5}


@pytest.mark.parametrize("strategy, param", integer_params(flags=False))
def test_no_strategy_accepts_a_window_of_zero(strategy, param):
    # 0 or a negative window used to crash deep in numpy or, worse, give a plausible but meaningless backtest
    with pytest.raises(ValueError, match=f"parameter {param}: 0 is too small"):
        build_algorithm(strategy, {}).with_params(**{param: 0})


@pytest.mark.parametrize("strategy, param", integer_params(flags=True))
def test_every_strategy_flag_can_be_off(strategy, param):
    assert build_algorithm(strategy, {}).with_params(**{param: 0}).values[param] == 0


# -- shares (0 to 1): a percentage is applied as what it says, an impossible share is refused ------------------------
SHARE = {"fraction": Param(0.5, 0.05, 1.0, share=True)}


@pytest.mark.parametrize("value, expected", [
    (0.5, 0.5),
    ("50%", 0.5),
    (" 50 % ", 0.5),
    (50, 0.5),  # above 1 a share can only mean percent
    (100, 1.0),
    (1, 1.0),  # 1 is the whole, not 1 %
    (0, 0.0),
    ("0.5%", 0.005),
    ("100%", 1.0),
])
def test_a_share_is_read_as_a_fraction_or_a_percentage(value, expected):
    assert resolve_params(SHARE, {"fraction": value})["fraction"] == pytest.approx(expected)


@pytest.mark.parametrize("value, reason", [
    (-0.1, "must be a share from 0 to 1"),
    (150, "must be a share from 0 to 1"),
    ("101%", "must be a share from 0 to 1"),
    ("-5%", "must be a share from 0 to 1"),
    ("abc%", "is not a number"),
    ("%", "is not a number"),
    ("nan%", "is not a finite number"),
])
def test_an_impossible_share_is_refused_by_name(value, reason):
    with pytest.raises(ValueError, match=f"parameter fraction: .*{reason}"):
        resolve_params(SHARE, {"fraction": value})


def test_a_percentage_for_a_parameter_that_is_no_share_is_refused():
    with pytest.raises(ValueError, match=r"parameter level: '30%' is not a number \(only shares"):
        resolve_params(SPACE, {"level": "30%"})


def test_a_number_above_one_for_a_parameter_that_is_no_share_stays_as_it_is():
    assert resolve_params(SPACE, {"level": 50})["level"] == 50.0


@pytest.mark.parametrize("model, param", [("fixed", "size.fraction"), ("risk", "size.risk"),
                                          ("kelly", "size.kelly_fraction")])
def test_the_sizing_shares_take_percentages(model, param):
    from tradebot.algorithms.sizing import sized

    assert sized(build_algorithm("sma_cross"), model, **{param: "50%"}).values[param] == 0.5


def test_the_volatility_target_is_no_share():
    # an annual volatility above 100 % is possible, so 1.2 must stay 1.2, not become 1.2 %
    from tradebot.algorithms.sizing import sized

    assert sized(build_algorithm("sma_cross"), "vol", **{"size.target_vol": 1.2}).values["size.target_vol"] == 1.2


def test_size_fraction_fifty_puts_half_the_capital_on():
    # the reported case: -p size.fraction=50 used to be capped to the whole capital
    from tradebot.algorithms.sizing import sized

    from .conftest import make_ohlc

    _, size = sized(build_algorithm("sma_cross"), "fixed", **{"size.fraction": 50}).run_sized(make_ohlc(300))
    assert set(size.unique()) == {0.5}


def test_a_take_profit_selling_half_in_percent():
    algo = build_algorithm("combo", {}, None, "rsi_threshold", "take_profit(5, 50)")
    assert algo.values["take_profit.size"] == 0.5


def test_the_vote_threshold_takes_a_percentage():
    assert build_algorithm("vote", {"threshold": "70%"}, ["rsi_threshold", "sma_cross"]).values["threshold"] == 0.7
