#!/usr/bin/env pwsh
# Test 9: Create and repair with 100 blocks

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "100blocks.tar.gz") -Destination "."

    Write-Banner "Create par2files using 100 blocks"

    # Get all files for par2 creation
    $files = Get-ChildItem -File | Select-Object -ExpandProperty Name
    $exitCode = Invoke-Par2 -Arguments (@("c", "-b100", "testdata.par2") + $files)
    if ($exitCode -ne 0) {
        Exit-TestWithError "Construction of files using PAR 2.0 failed"
    }

    Write-Banner "Repair 5% of 100 blocks par2files removing 3 files"

    # Remove 3 files
    Remove-Item "file" -ErrorAction SilentlyContinue
    Remove-Item "file1" -ErrorAction SilentlyContinue
    Remove-Item "file3" -ErrorAction SilentlyContinue

    # Get remaining files to pass as extra args
    $remainingFiles = Get-ChildItem -File | Where-Object { $_.Extension -ne ".par2" } | Select-Object -ExpandProperty Name
    $exitCode = Invoke-Par2 -Arguments (@("r", "testdata.par2") + $remainingFiles)
    if ($exitCode -ne 0) {
        Exit-TestWithError "Repair of files using PAR 2.0 failed (3 files removed)"
    }

    Write-Banner "Repair 5% of 100 blocks par2files removing 1 file"

    # Remove 1 more file
    Remove-Item "file5" -ErrorAction SilentlyContinue

    # Get remaining files to pass as extra args
    $remainingFiles = Get-ChildItem -File | Where-Object { $_.Extension -ne ".par2" } | Select-Object -ExpandProperty Name
    $exitCode = Invoke-Par2 -Arguments (@("r", "testdata.par2") + $remainingFiles)
    if ($exitCode -ne 0) {
        Exit-TestWithError "Repair of files using PAR 2.0 failed (1 file removed)"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
