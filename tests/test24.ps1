#!/usr/bin/env pwsh
# Test 24: par2 can be run from any starting dir (repair)

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    $faraway = Join-Path $PWD "in\a\folder\far\far\away"

    New-Item -ItemType Directory -Path $faraway -Force | Out-Null
    New-Item -ItemType Directory -Path "rundir" -Force | Out-Null

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination $faraway
    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata-par2files.tar.gz") -Destination $faraway

    # Save original files
    for ($i = 0; $i -le 9; $i++) {
        Copy-Item (Join-Path $faraway "test-$i.data") (Join-Path $faraway "test-$i.data.orig")
    }

    Write-Banner "par2 can be run from any starting dir"

    Set-Location "rundir"

    # Delete some files
    Remove-Item (Join-Path $faraway "test-1.data")
    Remove-Item (Join-Path $faraway "test-3.data")

    $exitCode = Invoke-Par2 -Arguments @("r", "-B", $faraway, (Join-Path $faraway "testdata.par2"))
    if ($exitCode -ne 0) {
        Exit-TestWithError "repair of PAR 2.0 files failed"
    }

    # Verify repaired files match originals
    if (-not (Compare-Files (Join-Path $faraway "test-1.data") (Join-Path $faraway "test-1.data.orig"))) {
        Exit-TestWithError "Repaired file test-1.data does not match original"
    }
    if (-not (Compare-Files (Join-Path $faraway "test-3.data") (Join-Path $faraway "test-3.data.orig"))) {
        Exit-TestWithError "Repaired file test-3.data does not match original"
    }

    Set-Location ..
    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}