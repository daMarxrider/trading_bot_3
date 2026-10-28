# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 daMarxrider
"""Entry point for the frozen (PyInstaller) binary."""

from tradebot.cli.main import app

if __name__ == "__main__":
    app(prog_name="tradebot")
