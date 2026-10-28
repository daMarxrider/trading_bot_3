# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Entry point for the windowed Windows binary (tradebot-gui.exe), which opens the GUI without a console."""

import os
import sys

# a windowed exe has no console, so sys.stdout/stderr are None and any library print would crash
for name in ("stdout", "stderr"):
    if getattr(sys, name) is None:
        setattr(sys, name, open(os.devnull, "w"))

from tradebot.gui.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
