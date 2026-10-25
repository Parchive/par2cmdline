#!/usr/bin/env pwsh
# Test 43: Reject repair through dangling symlink

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname
    $runDir = $PWD.Path

    Write-Banner "Reject repair through dangling symlink"

    New-Item -ItemType Directory -Path "work\.config\autostart" -Force | Out-Null
    New-Item -ItemType Directory -Path "outside" -Force | Out-Null
    [System.IO.File]::WriteAllText(
        (Join-Path $runDir "work\.config\autostart\poc.desktop"),
        "symlink escape file body`n")

    Set-Location (Join-Path $runDir "work")
    [System.IO.Directory]::SetCurrentDirectory($PWD.Path)

    $create = Invoke-Par2 -Arguments @("c", "-q", "-r100", "recovery.par2", ".config\autostart\poc.desktop") -ReturnObject
    if ($create.ExitCode -ne 0) {
        Exit-TestWithError "failed to create test recovery set"
    }

    Remove-Item ".config\autostart\poc.desktop" -Force

    New-Symlink -Path ".config\autostart\poc.desktop" -Target (Join-Path $runDir "outside\escaped-by-repair")

    $repair = Invoke-Par2 -Arguments @("r", "-q", "recovery.par2") -ReturnObject

    Set-Location $runDir
    [System.IO.Directory]::SetCurrentDirectory($PWD.Path)

    if (Test-Path -LiteralPath (Join-Path $runDir "outside\escaped-by-repair")) {
        Exit-TestWithError "repair wrote through dangling symlink outside repair tree"
    }

    if ($repair.ExitCode -eq 0) {
        Exit-TestWithError "repair through dangling symlink succeeded unexpectedly"
    }

    Write-Host "Dangling symlink repair target was rejected"

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
