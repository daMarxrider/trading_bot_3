# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""News sentiment (#167): a score per headline and an hourly score per symbol that lines up with the candles.

The score comes from a small word list for financial headlines (no model download, no key, works offline): every
word of the headline that is on the list adds its weight, a negation ("not", "no", "fails to") flips the next
words and an intensifier ("sharply", "record") strengthens them. The sum is squashed into ``-1 .. +1``: within
:data:`NEUTRAL` of 0 is neutral, above it positive, below it negative. It is a rough guide to the tone of the news,
not a forecast, and a headline with no known word scores 0.

:func:`history` turns the headlines of a symbol into hourly mean scores (``news_sentiment``, see
:mod:`tradebot.data.market`, which caches them and lines them up with the candles without look-ahead). Feeds only
serve their latest headlines, so the history is as long as the cache has been collecting.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

NEUTRAL = 0.15  #: a score within +-NEUTRAL counts as neutral

_POSITIVE = {
    1: "gain gains gained rise rises rose up higher climb climbs rally rallies rallied advance advances recover "
       "recovers recovery rebound rebounds rebounded beat beats upgrade upgraded upgrades grow grows growth strong "
       "stronger strength boost boosts boosted positive profit profits profitable optimism optimistic "
       "approve approved approval wins win won expands expansion partnership launch launches record inflow inflows "
       "accumulate accumulation support supports easing eases",
    2: "surge surges surged soar soars soared jump jumps jumped breakout outperform outperforms breakthrough "
       "adoption bull bullish upbeat milestone",
    3: "skyrocket skyrockets skyrocketed boom booms booming blowout",
}
_NEGATIVE = {
    1: "fall falls fell down lower drop drops dropped decline declines declined slide slides slid slip slips "
       "slipped weak weaker weakness cut cuts cutting miss misses missed downgrade downgraded downgrades loss "
       "losses lose loses lost fear fears worry worries worried concern concerns risk risks warn warns warning "
       "warned sell-off selloff outflow outflows delay delays delayed probe lawsuit sues sued fine fined ban bans "
       "banned halt halts halted bearish pessimism pessimistic layoffs recession inflation slump slumps",
    2: "plunge plunges plunged tumble tumbles tumbled sink sinks sank crash crashes crashed fraud hack hacked "
       "exploit exploited investigation default defaults bankrupt liquidation liquidations liquidated "
       "scandal collapses collapsed crisis",
    3: "collapse bankruptcy catastrophe disaster ponzi insolvent insolvency",
}
LEXICON: dict[str, int] = {}
for _weight, _words in _POSITIVE.items():
    LEXICON.update({w: _weight for w in _words.split()})
for _weight, _words in _NEGATIVE.items():
    LEXICON.update({w: -_weight for w in _words.split()})
NEGATIONS = frozenset("not no never without n't fails fail failed unable cannot cant won't wont isn't isnt "
                      "didn't didnt doesn't doesnt hasn't hasnt".split())
INTENSIFIERS = {"sharply": 1.5, "sharp": 1.5, "heavily": 1.5, "massive": 1.5, "huge": 1.5, "major": 1.3,
                "significantly": 1.4, "significant": 1.3, "record": 1.4, "biggest": 1.3, "slightly": 0.6,
                "modestly": 0.6, "modest": 0.6, "slight": 0.6}
_WORD = re.compile(r"[a-z][a-z'-]*", re.I)
NORMALISE = 4.0  #: the squashing constant: a sum of 3 scores about 0.83


def score_text(text: str) -> float:
    """Sentiment of ``text`` from -1 (very negative) to +1 (very positive); 0 when no known word is in it."""
    total, negate, boost = 0.0, 0, 1.0
    for word in (w.lower() for w in _WORD.findall(text or "")):
        if word in NEGATIONS:
            negate = 3  # a negation reaches the next three words
            continue
        if word in INTENSIFIERS:
            boost = INTENSIFIERS[word]
            continue
        weight = LEXICON.get(word)
        if weight is not None:
            total += weight * boost * (-0.8 if negate else 1.0)
            boost = 1.0
        negate = max(0, negate - 1)
    return total / math.sqrt(total * total + NORMALISE) if total else 0.0


def score_item(item) -> float:
    """Score of a :class:`~tradebot.news.NewsItem`: the headline counts most, its summary adds some."""
    head = score_text(item.title)
    summary = score_text(item.summary) if getattr(item, "summary", "") else 0.0
    return 0.75 * head + 0.25 * summary if summary else head


def label(score: float) -> str:
    return "positive" if score > NEUTRAL else "negative" if score < -NEUTRAL else "neutral"


def arrow(score: float) -> str:
    """The triangle that goes with a score (colour is never the only cue)."""
    return "▲" if score > NEUTRAL else "▼" if score < -NEUTRAL else "–"


@dataclass(frozen=True)
class Summary:
    """The headlines of one symbol, in one number."""

    count: int
    mean: float
    positive: int
    negative: int

    @property
    def neutral(self) -> int:
        return self.count - self.positive - self.negative


def summarize(scores: Iterable[float]) -> Summary:
    s = [float(x) for x in scores]
    return Summary(len(s), sum(s) / len(s) if s else 0.0, sum(x > NEUTRAL for x in s), sum(x < -NEUTRAL for x in s))


def _scored(items) -> pd.Series:
    rows = [(pd.Timestamp(n.published), score_item(n)) for n in items]
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([t for t, _ in rows])
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return pd.Series([v for _, v in rows], index=idx)


def hourly(items) -> pd.Series:
    """Mean headline score per hour (index: the hour's start, UTC) of ``items``, which should be of one symbol."""
    s = _scored(items)
    if s.empty:
        return pd.Series(dtype=float, name="news_sentiment")
    out = s.groupby(s.index.floor("h")).mean().sort_index()
    out.name = "news_sentiment"
    return out


def daily(items) -> pd.DataFrame:
    """``mean`` and ``count`` of the headline scores per UTC day, oldest first."""
    s = _scored(items)
    if s.empty:
        return pd.DataFrame({"mean": [], "count": []})
    g = s.groupby(s.index.floor("D"))
    return pd.DataFrame({"mean": g.mean(), "count": g.size()}).sort_index()


def history(symbol: str, fetch: Callable | None = None, limit: int = 200) -> pd.Series:
    """Hourly sentiment of ``symbol``'s latest headlines. The hour of the oldest headline is dropped when there are
    more hours, as the feed may have cut it off (it would otherwise overwrite a fuller, cached hour)."""
    from tradebot import news

    s = hourly((fetch or news.fetch_news)([symbol], limit))
    return s.iloc[1:] if len(s) > 1 else s
