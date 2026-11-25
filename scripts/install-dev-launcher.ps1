# SPDX-License-Identifier: GPL-3.0-or-later
# Adds a "Tradebot (dev)" Start Menu shortcut for this source checkout (#213).
#   scripts\install-dev-launcher.ps1              install
#   scripts\install-dev-launcher.ps1 -Uninstall   remove
param([switch]$Uninstall)
Set-Location "$PSScriptRoot\.."
$action = if ($Uninstall) { 'uninstall' } else { 'install' }
uv run --extra gui tradebot desktop $action --dev
exit $LASTEXITCODE
