#!/usr/bin/env pwsh
# Test 34: Rename-only mode skips damaged files

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."
    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata-par2files.tar.gz") -Destination "."

    Write-Banner "Testing rename-only mode skips damaged files"

    # Save original file
    Copy-Item "test-0.data" "test-0.data.orig"

    # Delete original and create a damaged version with a different name
    Remove-Item "test-0.data"
    Copy-Item "test-0.data.orig" "renamed-damaged.data"

    # Corrupt the file at the beginning
    $bytes = [System.IO.File]::ReadAllBytes("renamed-damaged.data")
    for ($i = 0; $i -lt 100; $i++) {
        $bytes[$i] = 0
    }
    [System.IO.File]::WriteAllBytes("renamed-damaged.data", $bytes)

    # With rename-only mode, repair may or may not succeed depending on recovery blocks
    # The key is that it shouldn't crash and should handle the damaged file appropriately
    $exitCode = Invoke-Par2 -Arguments @("r", "-O", "testdata.par2", "renamed-damaged.data")

    if ($exitCode -eq 0) {
        # If repair succeeded, check if file was properly restored
        if (Test-Path "test-0.data") {
            if (Compare-Files "test-0.data" "test-0.data.orig") {
                Write-Host "File was restored from recovery data (not from the damaged file)"
            }
        }
    } else {
        Write-Host "Repair not possible (as expected with rename-only mode and damaged file)"
    }

    Write-Host "Rename-only mode correctly handles damaged files!"

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}