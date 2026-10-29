# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Disk cache for candles and symbol lists (#63), in the OS cache folder (:func:`~tradebot.config.os_cache_dir`).

Layout::

    ohlc/<symbol>/<timeframe>.npz     candles of one series (timestamps + open/high/low/close/volume)
    symbols/*.json                    crypto pairs, Yahoo search results, instrument names

Everything here can be deleted at any time; it is rebuilt on demand. A file's mtime is its last use, so
:meth:`DiskCache.cleanup` can drop what wasn't opened for a while and, over the size cap, the least
recently used series first. ``[cache]`` in config.toml switches it off or sets the cap::

    [cache]
    enabled = true
    max_mb = 500
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.config import load_config, os_cache_dir

log = logging.getLogger(__name__)

DAY = 86400.0
COLUMNS = ("open", "high", "low", "close", "volume")
MAX_BARS = 50_000  #: newest candles kept per series
UNUSED_DAYS = 60  #: a series not opened for this long is deleted
UNUSED_DAYS_INTRADAY = 14  #: the same for 1m–30m candles: large, and Yahoo only serves 7–60 days of them
INTRADAY = {"1m", "3m", "5m", "15m", "30m"}
DEFAULT_MAX_MB = 500
CLEANUP_EVERY = DAY
LEGACY = ("crypto_pairs.json", "symbol_search.json")  #: symbol caches from before #63, in the folder's root


def _file_name(symbol: str) -> str:
    """``BTC/USDT`` -> ``BTC%2FUSDT``: safe on every OS, reversible, readable."""
    return re.sub(r"[^A-Za-z0-9._^=-]", lambda m: "".join(f"%{b:02X}" for b in m.group().encode()), symbol)


@dataclass
class CacheInfo:
    folder: Path
    ohlc_bytes: int
    symbol_bytes: int
    series: int

    @property
    def total_bytes(self) -> int:
        return self.ohlc_bytes + self.symbol_bytes


class DiskCache:
    def __init__(self, folder: Path | None = None, max_mb: float = DEFAULT_MAX_MB):
        self.folder = Path(folder) if folder is not None else os_cache_dir()
        self.max_bytes = int(max_mb * 1024 * 1024)

    @classmethod
    def from_config(cls) -> DiskCache | None:
        """The cache as config.toml's ``[cache]`` sets it up; None when it's switched off."""
        conf = load_config().get("cache", {})
        if not conf.get("enabled", True):
            return None
        try:
            max_mb = float(conf.get("max_mb", DEFAULT_MAX_MB))
        except (TypeError, ValueError):
            max_mb = DEFAULT_MAX_MB
        return cls(max_mb=max_mb)

    @property
    def ohlc_dir(self) -> Path:
        return self.folder / "ohlc"

    @property
    def symbols_dir(self) -> Path:
        return self.folder / "symbols"

    def path(self, symbol: str, timeframe: str) -> Path:
        return self.ohlc_dir / _file_name(symbol) / f"{timeframe}.npz"

    # -- candles ---------------------------------------------------------------------------------
    def load(self, symbol: str, timeframe: str) -> tuple[pd.DataFrame, pd.Timestamp | None] | None:
        """``(candles, listed)`` or None. ``listed``: no data exists before it (the listing date or the
        providers' history limit), so there's no point asking for older candles."""
        path = self.path(symbol, timeframe)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                unit = str(data["unit"]) if "unit" in data.files else "ns"
                index = pd.DatetimeIndex(data["time"].astype(f"datetime64[{unit}]")).tz_localize("UTC")
                df = pd.DataFrame({c: data[c] for c in COLUMNS if c in data.files}, index=index)
                listed = int(data["listed"]) if "listed" in data.files else -1
            os.utime(path)  # last use, for the cleanup
        except Exception as exc:  # corrupt or from an incompatible version: fetch again
            log.info("dropping unreadable cache file %s: %s", path, exc)
            path.unlink(missing_ok=True)
            return None
        return df, (pd.Timestamp(listed, tz="UTC") if listed >= 0 else None)

    def save(self, symbol: str, timeframe: str, df: pd.DataFrame, listed: pd.Timestamp | None = None) -> None:
        if df.empty:
            return
        df = df.iloc[-MAX_BARS:]
        path = self.path(symbol, timeframe)
        arrays = {"time": df.index.asi8, "unit": np.str_(df.index.unit),  # the providers' resolution, kept
                  "listed": np.int64(listed.value if listed is not None else -1)}
        arrays.update({c: df[c].to_numpy(dtype=float) for c in COLUMNS if c in df})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("wb") as fh:  # a file handle, so numpy doesn't append ".npz" to the temp name
                np.savez(fh, **arrays)
            tmp.replace(path)  # atomic: readers see the old or the new file, never half of one
        except OSError as exc:
            log.info("could not write cache file %s: %s", path, exc)

    def delete(self, symbol: str, timeframe: str) -> None:
        self.path(symbol, timeframe).unlink(missing_ok=True)

    # -- size and cleanup ------------------------------------------------------------------------
    def info(self) -> CacheInfo:
        series = list(self.ohlc_dir.glob("*/*.npz"))
        return CacheInfo(self.folder, sum(_size(p) for p in series),
                         sum(_size(p) for p in self.symbols_dir.glob("*")), len(series))

    def clear(self, ohlc: bool = True, symbols: bool = True) -> None:
        for wanted, folder in ((ohlc, self.ohlc_dir), (symbols, self.symbols_dir)):
            if wanted:
                shutil.rmtree(folder, ignore_errors=True)
        if symbols:
            for name in LEGACY:
                (self.folder / name).unlink(missing_ok=True)

    def cleanup(self, now: float | None = None) -> int:
        """Delete series unused for too long, then the least recently used ones while over the size cap
        (down to 80 % of it, so it doesn't run again right away). Returns the number of files deleted."""
        now = time.time() if now is None else now
        for name in LEGACY:  # moved to symbols/
            (self.folder / name).unlink(missing_ok=True)
        files = []
        for path in self.ohlc_dir.glob("*/*.npz"):
            try:
                st = path.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, path))
        deleted = 0
        keep = []
        for mtime, size, path in files:
            days = UNUSED_DAYS_INTRADAY if path.stem in INTRADAY else UNUSED_DAYS
            if now - mtime > days * DAY:
                path.unlink(missing_ok=True)
                deleted += 1
            else:
                keep.append((mtime, size, path))
        total = sum(size for _, size, _ in keep) + sum(_size(p) for p in self.symbols_dir.glob("*"))
        if total > self.max_bytes:
            for _, size, path in sorted(keep):  # oldest use first
                if total <= self.max_bytes * 0.8:
                    break
                path.unlink(missing_ok=True)
                total -= size
                deleted += 1
        for folder in self.ohlc_dir.glob("*"):  # symbols left without any series
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        return deleted

    def maybe_cleanup(self) -> int:
        """:meth:`cleanup` at most once a day (cheap to call at every start)."""
        stamp = self.folder / ".last_cleanup"
        try:
            if time.time() - stamp.stat().st_mtime < CLEANUP_EVERY:
                return 0
        except OSError:
            pass
        deleted = self.cleanup()
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            stamp.touch()
        except OSError:
            pass
        if deleted:
            log.info("cache cleanup deleted %d files", deleted)
        return deleted


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def human_size(n: int) -> str:
    """``1536`` -> ``1.5 KB``."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
