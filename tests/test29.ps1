#!/usr/bin/env pwsh
# Test 29: Issue 190, 1 bitflip can't be repaired

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "bug190.tar.gz") -Destination "."

    Write-Banner "Issue 190, 1 bitflip can't be repaired"

    # generate par2 from orig good file, copy that first
    Copy-Item "9MBones_crc_ok_orig" "9MBones_crc_ok"

    # Create PAR2 files
    $exitCode = Invoke-Par2 -Arguments @("c", "-m500", "-r30", "-n1", "-v", "9MBones_crc_ok")
    if ($exitCode -ne 0) {
        Exit-TestWithError "create failed"
    }

    # replace with bitflipped bad copy
    Copy-Item "9MBones_crc_ok_bad" "9MBones_crc_ok" -Force

    # Repair bitflip
    $exitCode = Invoke-Par2 -Arguments @("repair", "9MBones_crc_ok.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "bitflip repair failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}