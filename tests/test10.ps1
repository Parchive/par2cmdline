#!/usr/bin/env pwsh
# Test 10: Repair deleted subdir using PAR 2.0 data

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "smallsubdirdata.tar.gz") -Destination "."
    Expand-TarGz -Archive (Join-Path $TESTDATA "smallsubdirdata-par2files.tar.gz") -Destination "."

    Write-Banner "Repairing deleted subdir using PAR 2.0 data"

    Copy-Item "subdir1\test-0.data" "test-0.data.orig"
    Remove-Item -Recurse -Force "subdir1"

    $exitCode = Invoke-Par2 -Arguments @("r", "testdata.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Reconstruction of deleted subdir using PAR 2.0 failed"
    }

    if (-not (Compare-Files "subdir1\test-0.data" "test-0.data.orig")) {
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