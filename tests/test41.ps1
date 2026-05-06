#!/usr/bin/env pwsh
# Test 41: Take file list on stdin in combination with files on command-line

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "subdirdata.tar.gz") -Destination "."

    Write-Banner "Testing subdir data par2 creation with mixed files and stdin filelist"

    $exitCode = Get-Content (Join-Path $TESTDATA "subdirdata-partial-filelist.txt") | Invoke-Par2 -Arguments @("c", "recovery.par2", "subdir1\test-6.data", "subdir2\test-7.data", "@")
    if ($exitCode -ne 0) {
      Exit-TestWithError "failed to create parchive from stdin filelist"
    }

    $exitCode = Invoke-Par2 -Arguments @("r", "recovery.par2")
    if ($exitCode -ne 0) {
      Exit-TestWithError "failed to verify parchive created from stdin filelist"
    }

    Write-Host "Successfully created and verified par2 recovery with filelist from stdin"
    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
