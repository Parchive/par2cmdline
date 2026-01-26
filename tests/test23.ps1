#!/usr/bin/env pwsh
# Test 23: par2 can be run from any starting dir (verify)

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

    Write-Banner "par2 can be run from any starting dir"

    Set-Location "rundir"

    $exitCode = Invoke-Par2 -Arguments @("v", "-B", $faraway, (Join-Path $faraway "testdata.par2"))
    if ($exitCode -ne 0) {
        Exit-TestWithError "verify of PAR 2.0 files failed"
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