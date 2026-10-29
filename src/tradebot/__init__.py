# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""tradebot - extensible algorithmic trading platform.

Importing the package registers all built-in indicators, algorithms and brokers.
"""

__version__ = "1.1.0"

from tradebot import algorithms, brokers, indicators  # noqa: E402,F401  (populate registries)
