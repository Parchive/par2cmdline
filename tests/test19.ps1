#!/usr/bin/env pwsh
# Test 19: Skip leaway test

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "Creating 1024 byte test data file"

    New-RandomFile -Path "datafile" -SizeBytes 1024

    Write-Banner "Creating PAR 2.0 recovery data (block size 200)"

    $exitCode = Invoke-Par2 -Arguments @("c", "-s200", "-c1", "recovery", "datafile")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Creating PAR 2.0 data failed"
    }

    Write-Banner "Damaging data file (trim first 100 bytes)"

    Move-Item "datafile" "datafile.orig"
    $bytes = [System.IO.File]::ReadAllBytes("datafile.orig")
    $damagedBytes = $bytes[100..($bytes.Length - 1)]
    [System.IO.File]::WriteAllBytes("datafile", $damagedBytes)
    Remove-Item "datafile.orig"

    Write-Banner "Repairing using PAR 2.0 data (with skip leaway 99 - should fail)"

    $exitCode = Invoke-Par2 -Arguments @("r", "-N", "-S99", "-vv", "recovery.par2")
    if ($exitCode -eq 0) {
        Exit-TestWithError "Repair should not be possible with skip leaway set to 99"
    }

    Write-Banner "Repairing using PAR 2.0 data (with skip leaway 100 - should succeed)"

    $exitCode = Invoke-Par2 -Arguments @("r", "-N", "-S100", "-vv", "recovery.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Repair should be possible with skip leaway set to 100"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}