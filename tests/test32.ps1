#!/usr/bin/env pwsh
# Test 32: Bug 205, Files already exist so no par2 is created

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "subdirdata.tar.gz") -Destination "."

    Write-Banner "Bug 205, Files already exist so no par2 is created"

    New-Item -ItemType Directory -Path "par2" -Force | Out-Null

    # Get all items for recursive creation
    $allItems = Get-ChildItem | Select-Object -ExpandProperty Name
    $exitCode = Invoke-Par2 -Arguments (@("create", "-vv", "-a", "par2\disk1.par2", "-b32768", "-n31", "-R", "-v", "-B.\") + $allItems)
    if ($exitCode -ne 0) {
        Exit-TestWithError "Recursive creation of PAR 2.0 files failed"
    }

    $exitCode = Invoke-Par2 -Arguments @("verify", "-B.\", "par2\disk1.par2")
    if ($exitCode -ne 0) {
        Exit-TestWithError "verify failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}