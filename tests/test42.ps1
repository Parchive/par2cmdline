#!/usr/bin/env pwsh
# Test 42: Reject oversized PAR2 source block counts

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "block-count-wrap.tar.gz") -Destination "."

    Write-Banner "Rejecting oversized PAR2 source block counts"

    $result = Invoke-Par2 -Arguments @("r", "-q", "recovery.par2") -ReturnObject
    if ($result.ExitCode -eq 0) {
        Exit-TestWithError "Repair accepted oversized source block counts"
    }

    if ($result.StdErr -notmatch "Too many source blocks in recovery set") {
        Exit-TestWithError "Expected oversized block count rejection"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
