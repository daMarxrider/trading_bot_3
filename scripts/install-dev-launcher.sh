#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Adds a "Tradebot (dev)" entry for this source checkout to the app launcher (#213): a .desktop file on Linux, a
# launcher .app in ~/Applications on macOS (Windows: scripts/install-dev-launcher.ps1).
#   scripts/install-dev-launcher.sh              install
#   scripts/install-dev-launcher.sh --uninstall  remove
set -eu
cd "$(dirname "$0")/.."
case ${1:-} in
  --uninstall) exec uv run --extra gui tradebot desktop uninstall --dev ;;
  "") exec uv run --extra gui tradebot desktop install --dev ;;
  *) echo "usage: $0 [--uninstall]" >&2; exit 2 ;;
esac
