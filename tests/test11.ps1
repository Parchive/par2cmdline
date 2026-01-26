#!/usr/bin/env pwsh
# Test 11: Read beyond EOF handling

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "readbeyondeof.tar.gz") -Destination "."

    Write-Banner "Testing read beyond EOF handling"

    # The archive contains test.par2, not testdata.par2
    # This test should verify without error even with the edge case
    $exitCode = Invoke-Par2 -Arguments @("v", "test.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Read beyond EOF test failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
