#!/usr/bin/env pwsh
# Test 40: Take file list on stdin

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."

    Write-Banner "Testing flat data par2 creation with filelist from stdin"

    $exitCode = Get-Content (Join-Path $TESTDATA "flatdata-filelist.txt") | Invoke-Par2 -Arguments @("c", "recovery.par2", "@")
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
