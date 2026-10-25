#!/usr/bin/env pwsh
# Test 58: Reject a rename into place through a linked directory

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname
    $runDir = $PWD.Path

    Write-Banner "Reject a rename into place through a linked directory"

    New-Item -ItemType Directory -Path "base\sub" -Force | Out-Null
    New-Item -ItemType Directory -Path "outside" -Force | Out-Null
    [System.IO.File]::WriteAllText(
        (Join-Path $runDir "base\sub\data.bin"),
        "archive controlled content`n")

    $create = Invoke-Par2 -Arguments @("c", "-q", "-r100", "-Bbase", "base\recovery.par2", "base\sub\data.bin") -ReturnObject
    if ($create.ExitCode -ne 0) {
        Exit-TestWithError "could not create test recovery set"
    }

    # the target goes missing, a complete copy survives under another name, and
    # the directory it would be renamed into becomes a link pointing outside
    Move-Item "base\sub\data.bin" "base\copy.data"
    Remove-Item "base\sub"
    New-Symlink -Path "base\sub" -Target (Join-Path $runDir "outside")

    $repair = Invoke-Par2 -Arguments @("r", "-q", "-Bbase", "base\recovery.par2", "base\copy.data") -ReturnObject

    if (Test-Path -LiteralPath (Join-Path $runDir "outside\data.bin")) {
        Exit-TestWithError "a file was renamed through a linked directory outside the basepath"
    }

    if ($repair.ExitCode -eq 0) {
        Exit-TestWithError "rename into place through a linked directory succeeded unexpectedly"
    }

    Write-Host "Linked directory rename target was rejected"

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
