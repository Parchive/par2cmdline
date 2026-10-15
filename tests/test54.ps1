#!/usr/bin/env pwsh
# Test 54: Repairing from a complete copy which cannot be renamed into place

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."
    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata-par2files.tar.gz") -Destination "."

    Write-Banner "Repairing from a complete copy which cannot be renamed into place"

    Copy-Item "test-0.data" "test-0.data.orig"

    # A complete copy held open so that it can be read but not renamed
    New-Item -ItemType Directory -Path "donor" | Out-Null
    Move-Item "test-0.data" (Join-Path "donor" "copy.data")
    $held = [System.IO.File]::Open((Join-Path $PWD.Path "donor\copy.data"), [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)

    try {
        $result = Invoke-Par2 -Arguments @("r", "testdata.par2", "donor\copy.data") -ReturnObject
    }
    finally {
        $held.Close()
    }

    $output = $result.StdOut + $result.StdErr
    Write-Host $output

    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "Repair from a copy which cannot be renamed failed"
    }

    if ($output -notmatch "into place, so it will be copied") {
        Exit-TestWithError "The rename which failed was not reported"
    }

    if (-not (Compare-Files "test-0.data" "test-0.data.orig")) {
        Exit-TestWithError "Repaired file does not match the original"
    }

    if (-not (Compare-Files (Join-Path "donor" "copy.data") "test-0.data.orig")) {
        Exit-TestWithError "The copy was not left where it was"
    }

    Write-Host "Copy which cannot be renamed test passed!"

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
