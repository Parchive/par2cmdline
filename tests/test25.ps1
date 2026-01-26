#!/usr/bin/env pwsh
# Test 25: par2 with full path

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "par2 with full path"

    Set-Content -Path "some-file" -Value "file contents"
    Copy-Item "some-file" "some-file.orig"

    $fullPath = (Resolve-Path "some-file").Path
    $exitCode = Invoke-Par2 -Arguments @("create", $fullPath)
    if ($exitCode -ne 0) {
        Exit-TestWithError "Failed to create parchive"
    }

    Set-Content -Path "some-file" -Value "corrupted contents"

    $exitCode = Invoke-Par2 -Arguments @("repair", "some-file")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Failed to repair"
    }

    if (-not (Compare-Files "some-file" "some-file.orig")) {
        Exit-TestWithError "Repaired file does not match original"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}