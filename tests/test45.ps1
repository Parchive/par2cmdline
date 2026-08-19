#!/usr/bin/env pwsh
# Test 45: Scanning the blocks of a file in parallel finds the same data

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "Scanning the blocks of a file in parallel finds the same data"

    # The -T option only exists when built with thread support
    $help = Invoke-Par2 -Arguments @("-h") -ReturnObject
    if ($help.StdOut -notmatch "(?m)^  -T<n>") {
        Write-Host "Skipping: par2 was built without thread support."
        Complete-Test
        exit 77
    }

    # Build a data file of 64 blocks of 1024 bytes
    $builder = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt 1024; $i++) {
        [void]$builder.Append($i.ToString("D64"))
    }
    [System.IO.File]::WriteAllText((Join-Path $PWD "data.bin"), $builder.ToString())

    $create = Invoke-Par2 -Arguments @("c", "-q", "-s1024", "-c20", "test.par2", "data.bin") -ReturnObject
    if ($create.ExitCode -ne 0) {
        Exit-TestWithError "Could not create PAR2 files"
    }

    Copy-Item "data.bin" "data.bin.orig"

    $bytes = [System.IO.File]::ReadAllBytes((Join-Path $PWD "data.bin"))

    # A gap holding nothing: block 3 is corrupt
    for ($i = 0; $i -lt 16; $i++) {
        $bytes[3072 + $i] = [byte][char]'X'
    }

    # A gap holding data: shift blocks 10 to 19 forward by half a block, so that
    # they are still in the file but not where they are expected. Searching this
    # gap has to find them; only counting the blocks which were where they belong
    # would miss them.
    $region = New-Object byte[] 10752
    [System.Array]::Copy($bytes, 10240, $region, 0, 10752)
    [System.Array]::Copy($region, 0, $bytes, 10752, 10752)

    [System.IO.File]::WriteAllBytes((Join-Path $PWD "data.bin"), $bytes)

    # -T2 is the default and searches the whole file a byte at a time.
    # -T4 checks each block where it is expected first, and only searches
    # what that does not account for. Both must find the same data.
    $sequential = Invoke-Par2 -Arguments @("v", "-T2", "test.par2") -ReturnObject
    $parallel = Invoke-Par2 -Arguments @("v", "-T4", "test.par2") -ReturnObject

    if ($sequential.ExitCode -ne $parallel.ExitCode) {
        Exit-TestWithError "Exit code differed: -T2 gave $($sequential.ExitCode) and -T4 gave $($parallel.ExitCode)"
    }

    # Strip the progress indicator, which is rewritten in place with `r
    $sequentialBlocks = ($sequential.StdOut -replace "`r", "`n") -split "`n" | Where-Object { $_ -match "data blocks" }
    $parallelBlocks = ($parallel.StdOut -replace "`r", "`n") -split "`n" | Where-Object { $_ -match "data blocks" }

    if (($sequentialBlocks -join "`n") -ne ($parallelBlocks -join "`n")) {
        Write-Host "-T2:"; $sequentialBlocks | ForEach-Object { Write-Host "  $_" }
        Write-Host "-T4:"; $parallelBlocks | ForEach-Object { Write-Host "  $_" }
        Exit-TestWithError "Scanning the blocks in parallel found different data"
    }

    if (($parallelBlocks -join "`n") -notmatch "Found 62 of 64 data blocks") {
        Exit-TestWithError "Expected 62 of 64 blocks to be found"
    }

    # The repair must still work when the blocks were scanned in parallel
    $repair = Invoke-Par2 -Arguments @("r", "-q", "-T4", "test.par2") -ReturnObject
    if ($repair.ExitCode -ne 0) {
        Exit-TestWithError "Repair failed"
    }

    if (-not (Compare-Files -File1 "data.bin" -File2 "data.bin.orig")) {
        Exit-TestWithError "Repaired file does not match the original"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
