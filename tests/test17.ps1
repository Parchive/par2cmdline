#!/usr/bin/env pwsh
# Test 17: Remove subdir structure and repair (see #44)

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "bug44.tar.gz") -Destination "."

    Write-Banner "remove subdir structure and repair, see #44"

    Remove-Item -Recurse -Force "subdir1"

    $exitCode = Invoke-Par2 -Arguments @("r", "recovery.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "reparation of files using PAR 2.0 failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}