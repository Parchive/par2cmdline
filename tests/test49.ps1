#!/usr/bin/env pwsh
# Test 49: Checking the 16k hash of a file

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "hash16k-mismatch.tar.gz") -Destination "."

    Write-Banner "Checking the 16k hash of a file"

    # Every block of the file matches its verification entry, but the 16k hash
    # recorded for it does not. That costs 16k of hashing to check however
    # large the file is, so it is checked without being asked for.
    # A single thread leaves the whole file to the byte at a time search, and
    # several threads have the thread which reads the file hash it, so both
    # have to notice. The file is damaged rather than unreadable, so the exit
    # code is checked rather than just being non-zero, which an unrecognised
    # option would also give.
    foreach ($threads in $null, 1, 4) {
        $option = if ($null -eq $threads) { "" } else { " -t$threads" }

        $arguments = @("v", "-q")
        if ($null -ne $threads) { $arguments += "-t$threads" }
        $arguments += "recovery.par2"

        $verify = Invoke-Par2 -Arguments $arguments -ReturnObject
        if ($verify.ExitCode -ne 1) {
            Exit-TestWithError "verify$option gave $($verify.ExitCode), expected 1"
        }
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
