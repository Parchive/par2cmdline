#!/usr/bin/env pwsh
# Test 13: Repair file where 1 byte got removed at the end of a file

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."

    Write-Banner "create 5% recovery files"

    $exitCode = Invoke-Par2 -Arguments @("c", "testdata.par2", "test-0.data", "test-1.data", "test-2.data", "test-3.data", "test-4.data", "test-5.data", "test-6.data", "test-7.data", "test-8.data", "test-9.data")
    if ($exitCode -ne 0) {
        Exit-TestWithError "creating repair files using PAR 2.0 failed"
    }

    Write-Banner "repair files where 1 byte got removed at the end of a file"

    # Save original and truncate by 1 byte
    Move-Item "test-1.data" "test-1.data-correct"
    $bytes = [System.IO.File]::ReadAllBytes("test-1.data-correct")
    $truncatedBytes = $bytes[0..($bytes.Length - 2)]
    [System.IO.File]::WriteAllBytes("test-1.data", $truncatedBytes)

    $exitCode = Invoke-Par2 -Arguments @("r", "testdata.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "repair files using PAR 2.0 failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}