#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Adds tradebot to the app launcher from the extracted release folder (#213).
#   ./install.sh                 for the current user (~/.local/share)
#   sudo ./install.sh --system   for all users (/usr/local/share)
#   ./install.sh --uninstall     remove it again (add --system for a system-wide one)
# The entry points at this folder: run the script again after moving it.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
action=install
system=
for a in "$@"; do
  case $a in
    --uninstall) action=uninstall ;;
    --system) system=--system ;;
    -h|--help) sed -n '3,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done
exec "$here/tradebot" desktop "$action" $system
