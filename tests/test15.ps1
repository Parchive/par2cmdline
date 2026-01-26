#!/usr/bin/env pwsh
# Test 15: Repair files should succeed (issue #35)

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "par2-0.6.8-crash.tar.gz") -Destination "."

    Write-Banner "repair files should succeed, (issue #35)"

    Set-Location "par2-0.6.8-crash"

    $exitCode = Invoke-Par2 -Arguments @("r", "pack-ea5f7f848340980493ed39f5b7173d956c680e43.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "repair files using PAR 2.0 failed"
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