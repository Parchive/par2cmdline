#!/usr/bin/env pwsh
# Test 21: Save parfiles outside of basedirectory

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "save parfiles outside of basedirectory"

    New-Item -ItemType Directory -Path "parfiles" -Force | Out-Null
    New-Item -ItemType Directory -Path "datafiles" -Force | Out-Null

    Write-Banner "Creating PAR 2.0 recovery data"

    Set-Location "datafiles"
    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."

    $dataFiles = Get-ChildItem -Filter "*.data" | Select-Object -ExpandProperty Name
    $exitCode = Invoke-Par2 -Arguments (@("c", "-B", ".\", "..\parfiles\recovery") + $dataFiles)
    if ($exitCode -ne 0) {
        Exit-TestWithError "Creating PAR 2.0 data failed"
    }

    Write-Banner "Verifying PAR 2.0 recovery data"

    $exitCode = Invoke-Par2 -Arguments @("v", "-B", ".\", "..\parfiles\recovery.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Verifying PAR 2.0 data failed"
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