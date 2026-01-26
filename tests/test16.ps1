#!/usr/bin/env pwsh
# Test 16: Don't allow files outside par2 basedir (issue #34, #36)

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "subdirdata.tar.gz") -Destination "."

    Write-Banner "don't allow files outside par2 basedir, (issue #34, #36)"

    Set-Location "subdir1"

    $result = Invoke-Par2 -Arguments @("c", "-r2", "test.par2", "test-0.data", "test-1.data", "test-2.data", "test-3.data", "test-4.data", "..\subdir2\test-5.data", "..\subdir2\test-6.data", "..\subdir2\test-7.data", "..\subdir2\test-8.data", "..\subdir2\test-9.data") -ReturnObject
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "creating files using PAR 2.0 failed"
    }

    # Check if there were ignored files
    $output = $result.StdOut + $result.StdErr
    if ($output -notmatch "Ignoring") {
        Exit-TestWithError "there were no files ignored"
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