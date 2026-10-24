#!/usr/bin/env pwsh
# Test 56: Recursing into subdirectories for the files a wildcard matches

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "Recursing into subdirectories for the files a wildcard matches"

    New-Item -ItemType Directory -Path (Join-Path "sub" "deep") | Out-Null
    foreach ($file in @("top.txt", "sub\a.txt", "sub\deep\b.txt", "sub\notes.dat")) {
        Set-Content -Path $file -Value $file
    }

    # The wildcard matches files in the subdirectories as well as at the top.
    $create = Invoke-Par2 -Arguments @("c", "-q", "-q", "-R", "recursed.par2", "*.txt") -ReturnObject
    if ($create.ExitCode -ne 0) {
        Exit-TestWithError "Could not create with -R"
    }

    $verify = Invoke-Par2 -Arguments @("v", "recursed.par2") -ReturnObject
    $found = [regex]::Match($verify.StdOut, "There are (\d+) recoverable files").Groups[1].Value
    if ($found -ne "3") {
        Exit-TestWithError "-R with *.txt found $found files, expected 3"
    }

    # Without -R only the top is searched.
    $create = Invoke-Par2 -Arguments @("c", "-q", "-q", "flat.par2", "*.txt") -ReturnObject
    if ($create.ExitCode -ne 0) {
        Exit-TestWithError "Could not create without -R"
    }

    $verify = Invoke-Par2 -Arguments @("v", "flat.par2") -ReturnObject
    $found = [regex]::Match($verify.StdOut, "There are (\d+) recoverable files").Groups[1].Value
    if ($found -ne "1") {
        Exit-TestWithError "*.txt without -R found $found files, expected 1"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
