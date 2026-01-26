#!/usr/bin/env pwsh
# Test 3: Repair one missing file using PAR 2.0 data

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."
    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata-par2files.tar.gz") -Destination "."

    Write-Banner "Repairing one missing file using PAR 2.0 data"

    Copy-Item "test-0.data" "test-0.data.orig"
    Remove-Item "test-0.data"

    $exitCode = Invoke-Par2 -Arguments @("r", "testdata.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Reconstruction of one file using PAR 2.0 failed"
    }

    if (-not (Compare-Files "test-0.data" "test-0.data.orig")) {
        Exit-TestWithError "Repaired files do not match originals"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}