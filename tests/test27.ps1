#!/usr/bin/env pwsh
# Test 27: par2 with full path through symlink
# DUMMY script on windows, just here for completeness sake

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

Initialize-Test -TestName $testname

Write-Banner "par2 with full path through symlink"
Write-Host "SKIPPED: no symbolic link testing on Windows"

Complete-Test
exit 0
