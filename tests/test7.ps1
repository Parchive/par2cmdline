#!/usr/bin/env pwsh
# Test 7: Repair two files in subdirs using PAR 2.0 data generated on Windows

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "subdirdata.tar.gz") -Destination "."
    Expand-TarGz -Archive (Join-Path $TESTDATA "subdirdata-par2files-win.tar.gz") -Destination "."

    Write-Banner "Repairing two files in subdirs using PAR 2.0 data generated on Windows"

    Copy-Item "subdir1\test-2.data" "subdir1\test-2.data.orig"
    Copy-Item "subdir2\test-7.data" "subdir2\test-7.data.orig"

    Remove-Item "subdir1\test-2.data"
    Remove-Item "subdir2\test-7.data"

    $exitCode = Invoke-Par2 -Arguments @("r", "testdata.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Reconstruction of two files using PAR 2.0 failed"
    }

    if (-not (Compare-Files "subdir1\test-2.data" "subdir1\test-2.data.orig")) {
        Exit-TestWithError "Repaired file subdir1\test-2.data does not match original"
    }
    if (-not (Compare-Files "subdir2\test-7.data" "subdir2\test-7.data.orig")) {
        Exit-TestWithError "Repaired file subdir2\test-7.data does not match original"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}