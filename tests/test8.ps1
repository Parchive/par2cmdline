#!/usr/bin/env pwsh
# Test 8: Create PAR 2.0 data for files in subdirs, verify after moving folder

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "subdirdata.tar.gz") -Destination "."

    # Create source1 directory and move subdirs into it (matching Unix test)
    New-Item -ItemType Directory -Path "source1" -Force | Out-Null
    Move-Item "subdir1" "source1\"
    Move-Item "subdir2" "source1\"

    Push-Location "source1"

    Write-Banner "Create par2files on subdir"

    # Use wildcard equivalent - get all items in current directory
    $items = Get-ChildItem | Select-Object -ExpandProperty Name
    $exitCode = Invoke-Par2 -Arguments (@("c", "-R", "testdata.par2") + $items)
    if ($exitCode -ne 0) {
        Pop-Location
        Exit-TestWithError "Construction of files using PAR 2.0 failed"
    }

    Pop-Location

    # Rename source1 to source2 (simulating moving the source folder)
    Rename-Item "source1" "source2"

    Push-Location "source2"

    Write-Banner "Verify par2files on subdir, moved source folder"

    $exitCode = Invoke-Par2 -Arguments @("v", "testdata.par2")
    if ($exitCode -ne 0) {
        Pop-Location
        Exit-TestWithError "Verification of files using PAR 2.0 failed"
    }

    Pop-Location

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}