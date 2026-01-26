#!/usr/bin/env pwsh
# Test 2: Verify using PAR 2.0 data

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."
    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata-par2files.tar.gz") -Destination "."

    Write-Banner "Verifying using PAR 2.0 data"

    $exitCode = Invoke-Par2 -Arguments @("v", "testdata.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Initial PAR 2.0 verification failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}