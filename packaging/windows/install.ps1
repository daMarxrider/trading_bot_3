# SPDX-License-Identifier: GPL-3.0-or-later
# Adds tradebot to the Start Menu from the extracted release folder (#213).
#   .\install.ps1              for the current user
#   .\install.ps1 -System      for all users (run PowerShell as administrator)
#   .\install.ps1 -Uninstall   remove the shortcut again (add -System for the all-users one)
# The shortcut points at this folder: run the script again after moving it.
param([switch]$Uninstall, [switch]$System)
$action = if ($Uninstall) { 'uninstall' } else { 'install' }
$extra = if ($System) { @('--system') } else { @() }
& "$PSScriptRoot\tradebot.exe" desktop $action @extra
exit $LASTEXITCODE
