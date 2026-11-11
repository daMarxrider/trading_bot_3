# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Numbers for a pair of assets that move together (#117): z-score of the gap, half-life, hedge ratio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PairStats:
    bars: int
    correlation: float  #: of the legs' returns
    beta: float  #: hedge ratio: log a ≈ α + β · log b
    zscore: float  #: where the log spread is now, in standard deviations from its mean
    half_life: float | None  #: bars for a gap to close halfway; None when the spread doesn't revert
    spread_mean: float
    spread_std: float


def half_life(x: np.ndarray) -> float | None:
    """Mean-reversion half-life in bars from an AR(1) fit ``x[t] = c + φ·x[t-1]``: ``-ln 2 / ln φ``.
    None unless ``0 < φ < 1``, i.e. the series wanders off instead of coming back."""
    if len(x) < 10:
        return None
    phi = np.polyfit(x[:-1], x[1:], 1)[0]
    return float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else None


def pair_stats(a: pd.Series, b: pd.Series) -> PairStats:
    """Statistics of two close-price series, on the timestamps both have."""
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    df = df[(df > 0).all(axis=1)]
    if len(df) < 10:
        raise ValueError("need at least 10 common candles")
    la, lb = np.log(df.iloc[:, 0].to_numpy()), np.log(df.iloc[:, 1].to_numpy())
    beta = float(np.polyfit(lb, la, 1)[0])
    spread = la - beta * lb
    std = float(spread.std())
    return PairStats(
        bars=len(df),
        correlation=float(np.corrcoef(np.diff(la), np.diff(lb))[0, 1]),
        beta=beta,
        zscore=float((spread[-1] - spread.mean()) / std) if std > 0 else 0.0,
        half_life=half_life(spread),
        spread_mean=float(spread.mean()),
        spread_std=std,
    )


# -- rolling numbers: every value uses only candles before its bar, so a backtest can trade on them ----------
@dataclass
class Rolling:
    beta: pd.Series  #: hedge ratio from the last ``window`` candles before each bar
    correlation: pd.Series  #: of log prices over that window
    zscore: pd.Series  #: the bar's log spread against the window's regression line, in residual std


def _logs(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    df = df[(df > 0).all(axis=1)]
    return np.log(df.iloc[:, 0]), np.log(df.iloc[:, 1])


def rolling_pair(a: pd.Series, b: pd.Series, window: int = 60) -> Rolling:
    """Rolling OLS hedge ratio, correlation and z-score of the gap (NaN until ``window`` candles exist).

    β and the regression line come from the window ending at the *previous* bar, so the bar being judged
    never shapes its own yardstick. A short window makes β follow the pair but also absorbs the gap
    that is to be traded; a long one reacts slowly to a changed relationship."""
    la, lb = _logs(a, b)
    roll = dict(window=window, min_periods=window)
    var_b = lb.rolling(**roll).var()
    beta = la.rolling(**roll).cov(lb) / var_b
    alpha = la.rolling(**roll).mean() - beta * lb.rolling(**roll).mean()
    corr = la.rolling(**roll).corr(lb)
    resid_std = la.rolling(**roll).std() * np.sqrt((1 - corr**2).clip(lower=1e-12))
    beta, alpha, corr, resid_std = (s.shift(1) for s in (beta, alpha, corr, resid_std))
    z = (la - alpha - beta * lb) / resid_std
    return Rolling(beta, corr, z)


def kalman_pair(a: pd.Series, b: pd.Series, delta: float = 1e-4, obs_var: float = 1e-3, span: int = 100) -> Rolling:
    """Hedge ratio and z-score from a Kalman filter: ``log a = α + β·log b`` with (α, β) drifting as a random walk.

    Unlike :func:`rolling_pair` there is no window for β: it follows the pair smoothly instead of jumping when
    an old candle leaves the window. ``delta`` is how fast (α, β) may drift (larger: reacts faster, but
    absorbs the gap that is to be traded), ``obs_var`` the assumed noise of the gap (sets how much one candle
    moves the state). The z-score is each bar's forecast error from the state *before* seeing the bar, in
    units of the recent forecast errors (exponentially weighted over ``span`` candles, up to the previous
    bar), so a bar never shapes its own yardstick (no look-ahead). The first ~30 candles are NaN while the
    filter settles. ``correlation`` is the rolling one over 60 candles."""
    la, lb = _logs(a, b)
    y, x = la.to_numpy(), lb.to_numpy()
    n = len(y)
    state, cov = np.array([1.0, 0.0]), np.eye(2)  # (β, α)
    drift = delta / (1 - delta)
    lam, warm = 1 - 2 / (span + 1), 30
    beta, z = np.full(n, np.nan), np.full(n, np.nan)
    errs = np.zeros(n)
    var = 0.0
    for t in range(n):
        h = np.array([x[t], 1.0])
        cov = cov + drift * np.eye(2)  # predict
        err = y[t] - h @ state
        errs[t] = err
        if t == warm:
            var = float(np.mean(errs[:warm] ** 2))
        if t >= warm:
            beta[t] = state[0]
            z[t] = err / np.sqrt(var) if var > 0 else np.nan
            var = lam * var + (1 - lam) * err**2
        s_var = h @ cov @ h + obs_var  # update
        gain = cov @ h / s_var
        state = state + gain * err
        cov = cov - np.outer(gain, h) @ cov
    corr = la.rolling(60, min_periods=60).corr(lb).shift(1)
    return Rolling(pd.Series(beta, la.index), corr, pd.Series(z, la.index))


# -- cointegration: does the gap come back at all? ----------------------------------------------------------
#: Engle-Granger critical values for two series with a constant (MacKinnon), as {significance level: t value}
CRITICAL = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


@dataclass
class Cointegration:
    stat: float  #: ADF t value of the residual; the more negative, the more the gap comes back
    level: float  #: smallest significance level in CRITICAL the test passes (0.01 / 0.05 / 0.10), else 1.0
    beta: float

    @property
    def passed(self) -> bool:
        return self.level < 1.0

    @property
    def p(self) -> float:
        return p_value(self.stat)

    @property
    def p_text(self) -> str:
        return f"p < {self.level:g}" if self.passed else f"p > {max(CRITICAL):g}"


def p_value(stat: float) -> float:
    """Approximate p-value of an Engle-Granger t value: log-linear interpolation between the three critical
    values in :data:`CRITICAL` (extended past 1 % with the last slope, 1.0 beyond 10 %). Good enough to rank
    pairs and to correct for how many were tested; not a substitute for the exact MacKinnon tables."""
    pts = sorted(CRITICAL.items(), key=lambda kv: kv[1])  # (level, t) from most to least negative t, i.e. 0.01 first
    (p0, t0), (p1, t1) = pts[0], pts[1]
    if stat >= pts[-1][1]:
        return 1.0
    for (pa, ta), (pb, tb) in zip(pts, pts[1:]):
        if ta <= stat <= tb:
            return float(np.exp(np.log(pa) + (stat - ta) / (tb - ta) * (np.log(pb) - np.log(pa))))
    slope = (np.log(p1) - np.log(p0)) / (t1 - t0)
    return float(max(np.exp(np.log(p0) + (stat - t0) * slope), 1e-12))


def adf_stat(x: np.ndarray, lags: int | None = None) -> float:
    """Augmented Dickey-Fuller t value on ``x`` (no constant: for regression residuals). ``lags`` default
    follows Schwert's rule, capped so short series keep enough rows."""
    n = len(x)
    lags = min(int(12 * (n / 100) ** 0.25), max(0, n // 5)) if lags is None else lags
    dx = np.diff(x)
    rows = len(dx) - lags
    if rows < 10:
        raise ValueError("series too short for the test")
    y = dx[lags:]
    cols = [x[lags:-1]] + [dx[lags - i: len(dx) - i] for i in range(1, lags + 1)]
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (rows - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    return float(coef[0] / se)


def cointegration(a: pd.Series, b: pd.Series) -> Cointegration:
    """Engle-Granger test: regress log a on log b, then test the residual for a unit root (a gap that
    wanders off forever). Levels are coarse (1 / 5 / 10 %): enough to rank pairs, and honest about how few
    pairs a large scan lets through by chance."""
    la, lb = _logs(a, b)
    if len(la) < 30:
        raise ValueError("need at least 30 common candles")
    beta, alpha = np.polyfit(lb.to_numpy(), la.to_numpy(), 1)
    stat = adf_stat(la.to_numpy() - alpha - beta * lb.to_numpy())
    level = next((lv for lv in sorted(CRITICAL) if stat < CRITICAL[lv]), 1.0)
    return Cointegration(stat, level, float(beta))


# -- has the pair stopped working? --------------------------------------------------------------------------
@dataclass
class Health:
    broken: bool
    reasons: list[str]


def pair_health(a: pd.Series, b: pd.Series, recent: int = 60) -> Health:
    """Compare the last ``recent`` candles with the history before them, using the history's hedge ratio:
    the gap far outside what it used to be, stuck on one side of its mean for too long, or the legs
    decoupling all mean the gap may never close (merger, new business, regime change)."""
    la, lb = _logs(a, b)
    if len(la) < recent + 60:
        raise ValueError(f"need at least {recent + 60} common candles")
    old, new = slice(0, len(la) - recent), slice(len(la) - recent, None)
    beta = float(np.polyfit(lb.to_numpy()[old], la.to_numpy()[old], 1)[0])
    spread = (la - beta * lb).to_numpy()
    mean, std = spread[old].mean(), spread[old].std()
    reasons = []
    z = (spread[new] - mean) / std if std > 0 else np.zeros(recent)
    if abs(z).max() > 4:
        reasons.append(f"the gap reached {z[np.abs(z).argmax()]:+.1f} σ, beyond anything before")
    hl = half_life(spread[old])
    stuck = recent if (z > 0).all() or (z < 0).all() else 0
    if stuck and hl is not None and stuck > 3 * hl:
        reasons.append(f"the gap stayed on one side of its mean for {stuck} candles (half-life {hl:.0f})")
    ra, rb = np.diff(la.to_numpy()), np.diff(lb.to_numpy())
    before, now = np.corrcoef(ra[:-recent], rb[:-recent])[0, 1], np.corrcoef(ra[-recent:], rb[-recent:])[0, 1]
    if now < before - 0.3:
        reasons.append(f"return correlation fell from {before:.2f} to {now:.2f}")
    return Health(bool(reasons), reasons)


# -- finding pairs ------------------------------------------------------------------------------------------
@dataclass
class Candidate:
    a: str
    b: str
    correlation: float  #: of log returns, training part
    train: Cointegration
    test: Cointegration  #: the same test on the part the selection never saw
    beta: float
    half_life: float | None  #: on the test part, with the training β
    zscore: float  #: the gap now, against the training part's mean and std

    @property
    def symbol(self) -> str:
        return f"{self.a}:{self.b}"


@dataclass
class Scan:
    tested: int  #: pairs that passed the pre-filters and were tested
    skipped: int  #: pairs dropped by the pre-filters (correlation, group)
    passed_train: int  #: pairs the training test let through (after the correction)
    expected_by_chance: float  #: pairs a 5 % test passes with no real pairs at all: tested × 0.05
    found: list[Candidate]  #: passed the training test AND held up on the test part


CORRECTIONS = ("fdr", "bonferroni", "none")


def significant(pvalues: list[float], correction: str, level: float) -> list[bool]:
    """Which of ``pvalues`` count as significant at ``level`` after correcting for how many were tested:
    ``bonferroni`` (``p ≤ level / n``: no more than ``level`` false pair expected in the whole scan),
    ``fdr`` (Benjamini-Hochberg: about ``level`` of the pairs kept are false) or ``none`` (``p ≤ level``)."""
    n = len(pvalues)
    if correction == "none" or n == 0:
        return [p <= level for p in pvalues]
    if correction == "bonferroni":
        return [p <= level / n for p in pvalues]
    if correction == "fdr":
        order = sorted(range(n), key=pvalues.__getitem__)
        cut = max((k for k, i in enumerate(order, 1) if pvalues[i] <= level * k / n), default=0)
        keep = set(order[:cut])
        return [i in keep for i in range(n)]
    raise ValueError(f"correction must be one of {', '.join(CORRECTIONS)}")


def find_pairs(closes: dict[str, pd.Series], split: float = 0.6, min_correlation: float = 0.7,
               train_level: float = 0.05, test_level: float = 0.10, half_life_range: tuple[float, float] = (2, 60),
               correction: str = "fdr", groups: dict[str, str] | None = None) -> Scan:
    """Search ``closes`` (symbol -> close prices) for cointegrated pairs without fooling itself.

    The history is split in time. Pairs are picked on the first part only. Pre-filters keep the number of
    tests down: ``groups`` (symbol -> sector or any label; only symbols with the same label are paired, a
    symbol without a label is never paired) and ``min_correlation`` of the returns. Every remaining pair gets
    a cointegration test at ``train_level``, **corrected for the number of pairs tested** (``correction``,
    see :func:`significant`): without it, thousands of tests let some pass by chance, shown as
    ``expected_by_chance``. A pair counts as found only if it also holds up on the later part it wasn't
    picked on: cointegrated there at ``test_level`` and with a half-life in ``half_life_range``."""
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {', '.join(CORRECTIONS)}")
    frame = pd.concat(closes, axis=1, join="inner").dropna()
    frame = frame[(frame > 0).all(axis=1)]
    if len(frame) < 120:
        raise ValueError("need at least 120 candles all symbols share")
    cut = int(len(frame) * split)
    train, test = frame.iloc[:cut], frame.iloc[cut:]
    rets = np.log(train).diff().dropna().corr()
    names = list(frame.columns)
    skipped = 0
    tested: list[tuple[str, str, float, Cointegration]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = groups is None or (groups.get(a) is not None and groups.get(a) == groups.get(b))
            corr = float(rets.loc[a, b])
            if not same or corr < min_correlation:
                skipped += 1
                continue
            tested.append((a, b, corr, cointegration(train[a], train[b])))
    keep = significant([t[3].p for t in tested], correction, train_level)
    passed = 0
    found: list[Candidate] = []
    for (a, b, corr, tr), ok in zip(tested, keep):
        if not ok:
            continue
        passed += 1
        te = cointegration(test[a], test[b])
        spread_tr = np.log(train[a]) - tr.beta * np.log(train[b])
        spread_te = (np.log(test[a]) - tr.beta * np.log(test[b])).to_numpy()
        hl = half_life(spread_te)
        if te.level > test_level or hl is None or not half_life_range[0] <= hl <= half_life_range[1]:
            continue
        std = float(spread_tr.std())
        now = (np.log(frame[a].iloc[-1]) - tr.beta * np.log(frame[b].iloc[-1]) - spread_tr.mean()) / std
        found.append(Candidate(a, b, corr, tr, te, tr.beta, hl, float(now)))
    found.sort(key=lambda c: c.test.stat)
    return Scan(len(tested), skipped, passed, len(tested) * 0.05, found)
