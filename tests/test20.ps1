#!/usr/bin/env pwsh
# Test 20: Generate datafile with 2000 random bytes and repair split file

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "generate datafile with 2000 random bytes"

    New-RandomFile -Path "myfile.dat" -SizeBytes 2000

    Write-Banner "Creating PAR 2.0 recovery data"

    $exitCode = Invoke-Par2 -Arguments @("c", "-s1000", "-c0", "recovery", "myfile.dat")
    if ($exitCode -ne 0) {
        Exit-TestWithError "Creating PAR 2.0 data failed"
    }

    Write-Banner "split files"

    $bytes = [System.IO.File]::ReadAllBytes("myfile.dat")
    [System.IO.File]::WriteAllBytes("myfile.dat.001", $bytes[0..999])
    [System.IO.File]::WriteAllBytes("myfile.dat.002", $bytes[1000..1999])

    Remove-Item "myfile.dat"

    Write-Banner "Repairing using PAR 2.0 data"

    $allFiles = Get-ChildItem -File | Select-Object -ExpandProperty Name
    $exitCode = Invoke-Par2 -Arguments (@("r", "recovery.par2") + $allFiles)
    if ($exitCode -ne 0) {
        Exit-TestWithError "PAR 2.0 repair failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}