#!/usr/bin/env pwsh
# Test 18: Create PAR 2.0 recovery data and repair truncated file

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Copy-Item (Join-Path $TESTDATA "flatdata.tar.gz") ".\"

    Write-Banner "Creating PAR 2.0 recovery data"

    $exitCode = Invoke-Par2 -Arguments @("c", "recovery", "flatdata.tar.gz")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Creating PAR 2.0 data failed"
    }

    Write-Banner "Verifying using PAR 2.0 data"

    # Truncate the file
    Move-Item "flatdata.tar.gz" "flatdata.tar.gz-orig"
    $bytes = [System.IO.File]::ReadAllBytes("flatdata.tar.gz-orig")
    $truncatedBytes = $bytes[0..1982]  # Keep first 1983 bytes
    [System.IO.File]::WriteAllBytes("flatdata.tar.gz", $truncatedBytes)
    Remove-Item "flatdata.tar.gz-orig"

    $allFiles = Get-ChildItem -File | Select-Object -ExpandProperty Name
    $exitCode = Invoke-Par2 -Arguments (@("r", "recovery.par2") + $allFiles)
    if ($exitCode -ne 0) {
        Exit-TestWithError "PAR 2.0 repair failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}