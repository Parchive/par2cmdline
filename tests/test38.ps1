#!/usr/bin/env pwsh
# Test 34: Rename-only mode skips damaged files

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "subdirdata.tar.gz") -Destination "."
    Copy-Item (Join-Path $TESTDATA "subdirdata-partial-filelist.txt") "."

    Write-Banner "Testing subdir data par2 creation with filelist form file"

    $exitCode = Invoke-Par2 -Arguments @("c", "recovery.par2", "subdir1\test-6.data", "subdir2\test-7.data", "@subdirdata-partial-filelist.txt", "subdir1/test-8.data", "subdir2/test-9.data")
    if ($exitCode -ne 0) {
      Exit-TestWithError "failed to create parchive from filelist"
    }

    $exitCode = Invoke-Par2 -Arguments @("r", "recovery.par2")
    if ($exitCode -ne 0) {
      Exit-TestWithError "failed to verify parchive created from filelist"
    }

    Write-Host "Sucessfully created and verified par2 recovery with filelist from file"
    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
