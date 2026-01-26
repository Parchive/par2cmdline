#!/usr/bin/env pwsh
# Test 22: par2 can be run from any starting dir (create)

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

    Write-Banner "par2 can be run from any starting dir"

    Set-Location "rundir"

    $dataFiles = Get-ChildItem -Path $faraway -Filter "*.data" | Select-Object -ExpandProperty FullName
    $exitCode = Invoke-Par2 -Arguments (@("c", "-r2", "-B", $faraway, (Join-Path $faraway "test.par2")) + $dataFiles)
    if ($exitCode -ne 0) {
        Exit-TestWithError "create of PAR 2.0 files failed"
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