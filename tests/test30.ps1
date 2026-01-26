#!/usr/bin/env pwsh
# Test 30: Issue 128, 0 byte files cause issue

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."
    Expand-TarGz -Archive (Join-Path $TESTDATA "bug128-parfiles.tar.gz") -Destination "."

    Write-Banner "Issue 128, 0 byte files cause issue"

    # Create a 0 byte file
    New-Item -ItemType File -Path "test-a.data" -Force | Out-Null

    # Verify with 0 byte file
    $exitCode = Invoke-Par2 -Arguments @("verify", "recovery.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "0 byte file verify failed"
    }

    Remove-Item "test-a.data"

    # Repair with 0 byte file missing
    $exitCode = Invoke-Par2 -Arguments @("repair", "recovery.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "0 byte file repair failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}