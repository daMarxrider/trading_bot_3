# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Backend-agnostic time-series primitives.

Every function takes NumPy arrays *or* PyTorch tensors shaped ``[..., T]``
(time on the last axis) and returns the same type. Indicators and algorithms
are written once against these helpers, so the very same strategy code runs
in a pandas backtest on the CPU and in a batched population evaluation on a
CUDA/ROCm GPU inside the optimizer.

Warm-up periods are filled with NaN, matching pandas' rolling semantics.
"""

from __future__ import annotations

import numpy as np

try:  # torch is an optional dependency (``tradebot[gpu]``)
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

NAN = float("nan")


def is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def asarray(x, like):
    """Convert ``x`` to the backend/device/dtype of ``like``."""
    if is_torch(like):
        return torch.as_tensor(x, dtype=like.dtype, device=like.device)
    return np.asarray(x, dtype=like.dtype)


def where(cond, a, b):
    if is_torch(cond):
        like = a if is_torch(a) else b if is_torch(b) else None
        if like is None:
            like = torch.empty((), dtype=torch.get_default_dtype(), device=cond.device)
        return torch.where(cond, asarray(a, like), asarray(b, like))
    return np.where(cond, a, b)


def isnan(x):
    return torch.isnan(x) if is_torch(x) else np.isnan(x)


def nan_to_num(x, value=0.0):
    return torch.nan_to_num(x, nan=value) if is_torch(x) else np.nan_to_num(x, nan=value)


def abs_(x):
    return x.abs() if is_torch(x) else np.abs(x)


def maximum(a, b):
    if is_torch(a) or is_torch(b):
        like = a if is_torch(a) else b
        return torch.maximum(asarray(a, like), asarray(b, like))
    return np.maximum(a, b)


def cumsum(x):
    return torch.cumsum(x, dim=-1) if is_torch(x) else np.cumsum(x, axis=-1)


def cumprod(x):
    return torch.cumprod(x, dim=-1) if is_torch(x) else np.cumprod(x, axis=-1)


def cummax(x):
    return torch.cummax(x, dim=-1).values if is_torch(x) else np.maximum.accumulate(x, axis=-1)


def concat(parts):
    return torch.cat(parts, dim=-1) if is_torch(parts[0]) else np.concatenate(parts, axis=-1)


def full(shape, value, like):
    if is_torch(like):
        return torch.full(shape, value, dtype=like.dtype, device=like.device)
    return np.full(shape, value, dtype=like.dtype)


def shift(x, n: int = 1, fill=NAN):
    """Shift along time: ``shift(x, 1)[t] == x[t-1]``."""
    if n == 0:
        return x
    T = x.shape[-1]
    n_abs = min(abs(n), T)
    pad = full((*x.shape[:-1], n_abs), fill, x)
    if n > 0:
        return concat([pad, x[..., : T - n_abs]])
    return concat([x[..., n_abs:], pad])


def window_sum(x, n: int):
    """Rolling sum over ``n`` samples; NaN during warm-up."""
    cs = cumsum(x)
    out = cs - shift(cs, n, fill=0.0)
    return where(_warmup_mask(x, n), NAN, out)


def _warmup_mask(x, n):
    T = x.shape[-1]
    if is_torch(x):
        return torch.arange(T, device=x.device) < n - 1
    return np.arange(T) < n - 1


def sma(x, n: int):
    """Simple moving average; any NaN inside the window yields NaN."""
    valid = ~isnan(x)
    total = window_sum(where(valid, x, 0.0), n)
    count = window_sum(where(valid, 1.0, 0.0) if not is_torch(x) else valid.to(x.dtype), n)
    return where(count == n, total / n, NAN)


def rolling_std(x, n: int):
    """Population (ddof=0) rolling standard deviation."""
    mean = sma(x, n)
    var = sma(x * x, n) - mean * mean
    var = maximum(var, 0.0)
    return var.sqrt() if is_torch(var) else np.sqrt(var)


def _rolling_reduce(x, n: int, op: str):
    T = x.shape[-1]
    if T < n:
        return full(x.shape, NAN, x)
    if is_torch(x):
        win = x.unfold(-1, n, 1)
        red = win.amax(-1) if op == "max" else win.amin(-1)
    else:
        win = np.lib.stride_tricks.sliding_window_view(x, n, axis=-1)
        red = win.max(-1) if op == "max" else win.min(-1)
    return concat([full((*x.shape[:-1], n - 1), NAN, x), red])


def rolling_max(x, n: int):
    return _rolling_reduce(x, n, "max")


def rolling_min(x, n: int):
    return _rolling_reduce(x, n, "min")


def ewm(x, alpha):
    """Exponential moving average, ``adjust=False`` semantics, seeded at the first non-NaN value.

    ``alpha`` may be a scalar or broadcastable array (e.g. one value per population member).
    Scalar ``alpha`` with at most leading NaNs (the common case) uses a chunked
    parallel scan; anything else falls back to a per-bar loop.
    """
    if isinstance(alpha, (int, float)):
        valid = ~isnan(x)
        n_valid = cumsum(valid.to(x.dtype) if is_torch(x) else valid.astype(x.dtype))
        started = n_valid > 0
        if not bool((started & ~valid).any()):
            # seeding y[t0] = x[t0] equals feeding x[t0]/alpha into the recurrence from a zero state
            first = valid & (n_valid == 1)
            x0 = where(valid, x, 0.0)
            x0 = where(first, x0 / alpha, x0)
            return where(started, _linear_scan(alpha * x0, 1.0 - alpha), NAN)
    out = []
    state = None
    for t in range(x.shape[-1]):
        xt = x[..., t]
        if state is None:
            state = xt
        else:
            nxt = alpha * xt + (1 - alpha) * state
            state = where(isnan(state), xt, where(isnan(xt), state, nxt))
        out.append(state)
    if is_torch(x):
        return torch.stack(out, dim=-1)
    return np.stack(out, axis=-1) if out else x.copy()


def _linear_scan(u, b: float, chunk: int = 64):
    """``y[t] = b * y[t-1] + u[t]`` with ``y[-1] = 0``, via block matmuls instead of a per-bar loop."""
    T = u.shape[-1]
    n_chunks = -(-T // chunk)
    pad = n_chunks * chunk - T
    if pad:
        u = concat([u, full((*u.shape[:-1], pad), 0.0, u)])
    blocks = u.reshape(*u.shape[:-1], n_chunks, chunk)
    k = np.arange(chunk)
    powers = b ** np.clip(k[:, None] - k[None, :], 0, None) * (k[:, None] >= k[None, :])  # L[k, j] = b^(k-j)
    decay = b ** (k + 1)  # weight of the previous chunk's last value
    powers, decay = asarray(powers, u), asarray(decay, u)
    local = blocks @ powers.T  # within-chunk response from a zero start
    carry = None
    outs = []
    for c in range(n_chunks):
        y = local[..., c, :] if carry is None else local[..., c, :] + carry[..., None] * decay
        carry = y[..., -1]
        outs.append(y)
    return concat(outs)[..., :T]


def ema(x, n: int):
    return ewm(x, 2.0 / (n + 1))


def wilder(x, n: int):
    """Wilder's smoothing (RMA), used by RSI and ATR."""
    return ewm(x, 1.0 / n)


def latch(enter, exit):
    """Turn entry/exit event masks into a 0/1 holding state.

    The state switches to 1 on ``enter`` and back to 0 on ``exit`` and is held in
    between (forward fill). When both fire on the same bar, ``exit`` wins.
    Fully vectorized (cummax trick) so it also runs on the GPU.
    """
    if is_torch(enter) or is_torch(exit):
        enter, exit = torch.broadcast_tensors(torch.as_tensor(enter), torch.as_tensor(exit))
        events = torch.where(exit, 0, torch.where(enter, 1, -1))
        idx = torch.arange(enter.shape[-1], device=enter.device).expand(enter.shape)
        last = torch.cummax(torch.where(events >= 0, idx, 0), dim=-1).values
        # bars before the first event have last == 0 with events[0] == -1 -> clamp to flat
        return torch.gather(events, -1, last).clamp(min=0).to(torch.float64)
    enter, exit = np.broadcast_arrays(enter, exit)
    events = np.where(exit, 0, np.where(enter, 1, -1))
    idx = np.broadcast_to(np.arange(enter.shape[-1]), enter.shape)
    last = np.maximum.accumulate(np.where(events >= 0, idx, 0), axis=-1)
    return np.take_along_axis(events, last, axis=-1).clip(min=0).astype(float)


def cross_above(a, b):
    """True on the bar where ``a`` crosses from <= ``b`` to > ``b``."""
    return (a > b) & (shift(a, 1) <= shift(b, 1))


def cross_below(a, b):
    return (a < b) & (shift(a, 1) >= shift(b, 1))
