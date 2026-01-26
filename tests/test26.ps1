#!/usr/bin/env pwsh
# Test 26: par2 with full path and creation and repair in subfolder

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "par2 with full path and creation and repair in subfolder"

    New-Item -ItemType Directory -Path "subfolder" -Force | Out-Null

    Set-Content -Path "subfolder\some-file" -Value "file contents"
    Copy-Item "subfolder\some-file" "subfolder\some-file.orig"

    $fullPath = (Resolve-Path "subfolder\some-file").Path
    $exitCode = Invoke-Par2 -Arguments @("create", $fullPath)
    if ($exitCode -ne 0) {
        Exit-TestWithError "Failed to create parchive"
    }

    Set-Content -Path "subfolder\some-file" -Value "corrupted contents"

    $exitCode = Invoke-Par2 -Arguments @("repair", "subfolder\some-file")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Failed to repair"
    }

    if (-not (Compare-Files "subfolder\some-file" "subfolder\some-file.orig")) {
        Exit-TestWithError "Repaired file does not match original"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}