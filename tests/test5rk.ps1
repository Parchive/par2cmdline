#!/usr/bin/env pwsh
# Test 5rk: Create 100% PAR 2.0 recovery data using -rk option
# This is a copy of test5 but using "-rk" instead of "-r100"

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."

    Write-Banner "Creating 100% PAR 2.0 recovery data"

    # About the "magic -rk1302":
    # par2 c -r100 creates PAR2 files of in sum 1302436 bytes, so do
    # we approximately (-rk1302).
    # To also test ISSUE-80 we also use "-n2".
    $exitCode = Invoke-Par2 -Arguments @("c", "-rk1302", "-b190", "-n2", "-a", "newtest", "test-0.data", "test-1.data", "test-2.data", "test-3.data", "test-4.data", "test-5.data", "test-6.data", "test-7.data", "test-8.data", "test-9.data")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Creating PAR 2.0 data failed"
    }

    # Check that -n2 created exactly 2 vol files (case-insensitive for Windows)
    $volFiles = Get-ChildItem -Filter "newtest.vol*par2" | Measure-Object
    if ($volFiles.Count -ne 2) {
        Exit-TestWithError "File count option -n2 did not work. Expected 2, got $($volFiles.Count)"
    }

    # Save originals
    for ($i = 0; $i -le 9; $i++) {
        Copy-Item "test-$i.data" "test-$i.data.orig"
    }

    # Delete all data files
    Remove-Item "test-*.data"

    # Verify deletion worked
    if (Test-Path "test-0.data") {
        Exit-TestWithError "File deletion did not work"
    }

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