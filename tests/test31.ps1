#!/usr/bin/env pwsh
# Test 31: Bug 150, Files in rootfolder are not used with recursion

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "smallsubdirdata.tar.gz") -Destination "."

    Write-Banner "Bug 150, Files in rootfolder are not used with recursion"

    # Create files before and after subdirs alphabetically
    Set-Content -Path "aaaa-test.data" -Value "file in rootpath"
    Set-Content -Path "test-a.data" -Value "another file in rootpath"

    # Get all items SORTED ALPHABETICALLY to match Unix shell glob behavior
    # This ensures aaaa-test.data comes first and becomes the par2 base name
    $allItems = Get-ChildItem | Select-Object -ExpandProperty Name | Sort-Object
    
    $exitCode = Invoke-Par2 -Arguments (@("create", "-R") + $allItems)
    if ($exitCode -ne 0) {
        Exit-TestWithError "Recursive creation of PAR 2.0 files failed"
    }

    # The first file in the alphabetically sorted list is used as parfilename
    # so it's not covered for repair - remove it
    Remove-Item "aaaa-test.data"

    # With alphabetical sorting, aaaa-test.data comes first, so the par2 file
    # will be named aaaa-test.data.par2
    $mainPar2 = "aaaa-test.data.par2"

    $exitCode = Invoke-Par2 -Arguments @("verify", $mainPar2)
    if ($exitCode -ne 0) {
        Exit-TestWithError "verify of rootpath inclusion failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
