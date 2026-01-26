#!/usr/bin/env pwsh
# Test 33: Rename-only mode with renamed files

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."
    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata-par2files.tar.gz") -Destination "."

    Write-Banner "Testing rename-only mode with renamed files"

    # Save original files
    Copy-Item "test-0.data" "test-0.data.orig"
    Copy-Item "test-1.data" "test-1.data.orig"
    Copy-Item "test-2.data" "test-2.data.orig"

    # Rename some files (simulate accidental renaming)
    Move-Item "test-0.data" "renamed-file-a.data"
    Move-Item "test-1.data" "renamed-file-b.data"
    Move-Item "test-2.data" "renamed-file-c.data"

    # Verify fails without the renamed files
    $result = Invoke-Par2 -Arguments @("v", "testdata.par2") -ReturnObject
    $output = $result.StdOut + $result.StdErr
    if ($output -notmatch "missing") {
        Exit-TestWithError "Verification should report missing files"
    }

    # Repair with rename-only mode, passing the renamed files as extra files
    $exitCode = Invoke-Par2 -Arguments @("r", "-O", "testdata.par2", "renamed-file-a.data", "renamed-file-b.data", "renamed-file-c.data")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Repair with rename-only mode failed"
    }

    # Check that the original files are restored with correct names
    if (-not (Compare-Files "test-0.data" "test-0.data.orig")) {
        Exit-TestWithError "Repaired file test-0.data does not match original"
    }
    if (-not (Compare-Files "test-1.data" "test-1.data.orig")) {
        Exit-TestWithError "Repaired file test-1.data does not match original"
    }
    if (-not (Compare-Files "test-2.data" "test-2.data.orig")) {
        Exit-TestWithError "Repaired file test-2.data does not match original"
    }

    Write-Host "Rename-only mode test passed!"

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}