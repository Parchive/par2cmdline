#!/usr/bin/env pwsh
# Test 48: Checking the hash of the whole of a file when asked to

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "full-hash-mismatch.tar.gz") -Destination "."

    Write-Banner "Checking the hash of the whole of a file when asked to"

    # Every block of the file matches its verification entry, so the blocks
    # alone say the file is intact.
    $blocks = Invoke-Par2 -Arguments @("v", "-q", "recovery.par2") -ReturnObject
    if ($blocks.ExitCode -ne 0) {
        Exit-TestWithError "Expected the file to verify when only its blocks are checked"
    }

    # The whole file hash recorded for it does not match, which only shows up
    # when the whole of the file is hashed as well.
    # A single thread leaves the whole file to the byte at a time search, and
    # several threads have the thread which reads the file hash it, so both
    # have to notice. The file is damaged rather than unreadable, so the exit
    # code is checked rather than just being non-zero, which an unrecognised
    # option would also give.
    foreach ($threads in $null, 1, 4) {
        $option = if ($null -eq $threads) { "" } else { " -t$threads" }

        $arguments = @("v", "-q")
        if ($null -ne $threads) { $arguments += "-t$threads" }
        $arguments += @("--full-hash", "recovery.par2")

        $whole = Invoke-Par2 -Arguments $arguments -ReturnObject
        if ($whole.ExitCode -ne 1) {
            Exit-TestWithError "--full-hash$option gave $($whole.ExitCode), expected 1"
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
