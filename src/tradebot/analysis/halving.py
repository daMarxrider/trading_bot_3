# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""The Bitcoin halving schedule and a forecast of the next halvings (#99).

Bitcoin's rules fix the schedule in blocks, not in time. Two constants in ``src/kernel/chainparams.cpp``
(mainnet) define it:

* ``consensus.nSubsidyHalvingInterval = 210000``: the block reward halves every 210,000 blocks
* ``consensus.nPowTargetSpacing = 10 * 60``: the target time per block is 600 seconds

That gives the "4-year rule": 210,000 x 600 s = 1458.33 days. The difficulty only adjusts every 2016 blocks
and after the fact, so while the hashrate grows, blocks come faster than 600 s and every halving so far came
early (by 1.3 % to 9.5 %).

The forecast of the next halving is the last one plus 210,000 blocks at an estimated time per block. The
estimate comes from the completed epochs (genesis -> halving 1 -> halving 2 ...) using one of
:data:`ESTIMATORS`. :func:`backtest` predicts every past halving from the ones before it.
:data:`ESTIMATOR` is the estimator with the smallest error there (a test keeps it that way).

:func:`cycle` is the calendar clock used by the ``halving`` indicator and the ``halving_cycle`` strategy. On
every bar it only uses halvings that had happened by then, so a backtest never sees the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from tradebot.core import ops

INTERVAL = 210_000  #: blocks between halvings (nSubsidyHalvingInterval)
TARGET_SPACING = 600  #: target seconds per block (nPowTargetSpacing)
DAY = 86_400
NOMINAL_DAYS = INTERVAL * TARGET_SPACING / DAY  #: 1458.33, the "4-year rule"

GENESIS = 1_231_006_505  #: block 0, 2009-01-03 18:15:05 UTC
#: (block height, block timestamp in Unix seconds) of every halving so far (from the block headers)
HALVINGS: tuple[tuple[int, int], ...] = (
    (210_000, 1_354_116_278),  # 2012-11-28 15:24:38 UTC
    (420_000, 1_468_082_773),  # 2016-07-09 16:46:13 UTC
    (630_000, 1_589_225_023),  # 2020-05-11 19:23:43 UTC
    (840_000, 1_713_571_767),  # 2024-04-20 00:09:27 UTC
)
#: genesis followed by every halving: the start of each epoch
STARTS: tuple[int, ...] = (GENESIS, *(t for _, t in HALVINGS))


def _trend(spacings: Sequence[float]) -> float:
    """Straight line through the epochs' seconds per block, extended one epoch."""
    if len(spacings) < 2:
        return _mean(spacings)
    x = np.arange(len(spacings))
    slope, intercept = np.polyfit(x, spacings, 1)
    return float(slope * len(spacings) + intercept)


def _mean(spacings: Sequence[float]) -> float:
    return float(np.mean(spacings)) if len(spacings) else TARGET_SPACING


#: How the seconds per block of the next epoch are estimated from those of the completed epochs.
ESTIMATORS: dict[str, Callable[[Sequence[float]], float]] = {
    "nominal": lambda s: float(TARGET_SPACING),  # the 4-year rule
    "mean": _mean,  # average of all completed epochs
    "last": lambda s: float(s[-1]) if len(s) else TARGET_SPACING,  # the latest epoch
    "trend": _trend,  # linear trend over the epochs
}
ESTIMATOR = "mean"  #: the one with the smallest error predicting the past halvings (see :func:`best_estimator`)


def spacings(starts: Sequence[float]) -> list[float]:
    """Average seconds per block of each completed epoch, given the epoch starts."""
    return [(b - a) / INTERVAL for a, b in zip(starts, starts[1:])]


def predict(starts: Sequence[float] = STARTS, estimator: str = ESTIMATOR, count: int = 1) -> list[float]:
    """Unix timestamps of the next ``count`` halvings after the last of ``starts`` (genesis first)."""
    step = INTERVAL * ESTIMATORS[estimator](spacings(starts))
    return [starts[-1] + step * (i + 1) for i in range(count)]


@dataclass(frozen=True)
class Forecast:
    height: int
    actual: float
    predicted: float

    @property
    def error_days(self) -> float:
        """Predicted minus actual, in days (positive = predicted too late)."""
        return (self.predicted - self.actual) / DAY


def backtest(estimator: str) -> list[Forecast]:
    """Each past halving that has at least one earlier epoch, predicted from the halvings before it."""
    return [Forecast(HALVINGS[k - 1][0], STARTS[k], predict(STARTS[:k], estimator)[0])
            for k in range(2, len(STARTS))]


def mean_abs_error(estimator: str) -> float:
    return float(np.mean([abs(f.error_days) for f in backtest(estimator)]))


def best_estimator() -> str:
    return min(ESTIMATORS, key=mean_abs_error)


def next_from_tip(height: int, timestamp: float) -> tuple[int, float, float]:
    """``(height, timestamp, seconds per block)`` of the next halving extrapolated from a known block,
    at the pace of the current epoch so far (e.g. from a block explorer's chain tip)."""
    epoch_start = max(i for i, (h, _) in enumerate(((0, GENESIS), *HALVINGS)) if h <= height)
    start_height, start_time = ((0, GENESIS), *HALVINGS)[epoch_start]
    pace = (timestamp - start_time) / (height - start_height) if height > start_height else TARGET_SPACING
    target = (height // INTERVAL + 1) * INTERVAL
    return target, timestamp + (target - height) * pace, pace


TIP_URL = "https://blockstream.info/api/blocks/tip"


def fetch_tip(timeout: float = 15) -> tuple[int, int]:
    """``(height, timestamp)`` of the current chain tip from the Blockstream API (needs internet)."""
    import requests

    base = TIP_URL
    height = int(requests.get(f"{base}/height", timeout=timeout).text)
    block_hash = requests.get(f"{base}/hash", timeout=timeout).text.strip()
    block = requests.get(f"https://blockstream.info/api/block/{block_hash}", timeout=timeout).json()
    return int(block.get("height", height)), int(block["timestamp"])


def _segments(future: int = 3) -> list[tuple[float, float]]:
    """``(start, predicted next halving)`` per epoch, each predicted only from the epochs before it.

    The known epochs are followed by ``future`` predicted ones, so bars after a halving that isn't in
    :data:`HALVINGS` yet still get a cycle (from the forecast).
    """
    out = [(STARTS[k], predict(STARTS[:k + 1])[0]) for k in range(len(STARTS))]
    ahead = predict(STARTS, count=future + 1)
    out += list(zip(ahead[:-1], ahead[1:]))
    return out


def cycle(time) -> dict[str, object]:
    """The halving clock per bar for Unix-second timestamps ``time`` (NumPy or torch, any shape).

    * ``days``: days since the latest halving (since genesis before the first one)
    * ``days_to_next``: days until the next halving as it was predicted at that bar from the earlier
      halvings only; negative while a halving is later than predicted
    * ``phase``: ``days`` as a fraction of the predicted cycle length (0 at the halving, about 1 at the next)

    NaN before genesis and where ``time`` is NaN.
    """
    nan = ops.full(tuple(time.shape), np.nan, time)
    last, nxt = nan, nan
    for start, following in _segments():
        after = time >= start
        last = ops.where(after, float(start), last)
        nxt = ops.where(after, float(following), nxt)
    return {"days": (time - last) / DAY, "days_to_next": (nxt - time) / DAY, "phase": (time - last) / (nxt - last)}
