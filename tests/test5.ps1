#!/usr/bin/env pwsh
# Test 5: Create and repair 100% loss using PAR 2.0 data

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."

    Write-Banner "Creating 100% PAR 2.0 recovery data"

    $exitCode = Invoke-Par2 -Arguments @("c", "-r100", "-b190", "-a", "newtest", "test-0.data", "test-1.data", "test-2.data", "test-3.data", "test-4.data", "test-5.data", "test-6.data", "test-7.data", "test-8.data", "test-9.data")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Creating PAR 2.0 data failed"
    }

    # Save originals
    for ($i = 0; $i -le 9; $i++) {
        Copy-Item "test-$i.data" "test-$i.data.orig"
    }

    # Delete all data files
    Remove-Item "test-*.data"

    Write-Banner "Repairing 100% loss using PAR 2.0 data"

    $exitCode = Invoke-Par2 -Arguments @("r", "newtest.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Full Repair using PAR 2.0 failed"
    }

    # Verify all files
    for ($i = 0; $i -le 9; $i++) {
        if (-not (Compare-Files "test-$i.data" "test-$i.data.orig")) {
            Exit-TestWithError "Repaired file test-$i.data does not match original"
        }
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}