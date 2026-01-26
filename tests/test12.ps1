#!/usr/bin/env pwsh
# Test 12: par2-0.6.8 crash test

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "par2-0.6.8-crash.tar.gz") -Destination "."

    Write-Banner "Testing par2-0.6.8 crash case"

    # This test should not crash - exit code may vary
    $exitCode = Invoke-Par2 -Arguments @("v", "crashtest.par2")
    Write-Host "Crash test completed without crashing (exit code: $exitCode)"

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}