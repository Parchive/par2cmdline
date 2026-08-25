#!/usr/bin/env pwsh
# Test 47: Reading several files at once finds the same data

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "Reading several files at once finds the same data"

    # The -T option only exists when built with thread support
    $help = Invoke-Par2 -Arguments @("-h") -ReturnObject
    if ($help.StdOut -notmatch "(?m)^  -T<n>") {
        Write-Host "Skipping: par2 was built without thread support."
        Complete-Test
        exit 77
    }

    # Build four data files of 16 blocks of 1024 bytes. Each file holds
    # different data so that a block cannot be matched against the wrong file.
    foreach ($n in 1..4) {
        $builder = New-Object System.Text.StringBuilder
        for ($i = 0; $i -lt 256; $i++) {
            [void]$builder.Append($n.ToString() + $i.ToString("D63"))
        }
        [System.IO.File]::WriteAllText((Join-Path $PWD "data$n.bin"), $builder.ToString())
        Copy-Item "data$n.bin" "data$n.bin.orig"
    }

    $create = Invoke-Par2 -Arguments @("c", "-q", "-s1024", "-c20", "test.par2", "data1.bin", "data2.bin", "data3.bin", "data4.bin") -ReturnObject
    if ($create.ExitCode -ne 0) {
        Exit-TestWithError "Could not create PAR2 files"
    }

    # data1.bin is left alone, so it is matched entirely where it is expected.

    # data2.bin has a gap holding nothing: block 3 is corrupt.
    $bytes = [System.IO.File]::ReadAllBytes((Join-Path $PWD "data2.bin"))
    for ($i = 0; $i -lt 16; $i++) {
        $bytes[3072 + $i] = [byte][char]'X'
    }
    [System.IO.File]::WriteAllBytes((Join-Path $PWD "data2.bin"), $bytes)

    # data3.bin has a gap holding data: blocks 5 onwards are shifted forward by
    # half a block, so they are still in the file but not where they are expected.
    $bytes = [System.IO.File]::ReadAllBytes((Join-Path $PWD "data3.bin"))
    $region = New-Object byte[] 10240
    [System.Array]::Copy($bytes, 5120, $region, 0, 10240)
    [System.Array]::Copy($region, 0, $bytes, 5632, 10240)
    [System.IO.File]::WriteAllBytes((Join-Path $PWD "data3.bin"), $bytes)

    # data4.bin is truncated, so it cannot be a perfect match on length alone.
    $bytes = [System.IO.File]::ReadAllBytes((Join-Path $PWD "data4.bin"))
    $truncated = New-Object byte[] 12288
    [System.Array]::Copy($bytes, 0, $truncated, 0, 12288)
    [System.IO.File]::WriteAllBytes((Join-Path $PWD "data4.bin"), $truncated)

    # Reading one file at a time gives each file every thread for its blocks.
    # Reading four at once shares those threads out between them. Both must find
    # the same data. The files are reported in whatever order they finish, so the
    # lines are sorted before they are compared.
    $onefile = Invoke-Par2 -Arguments @("v", "-T1", "test.par2") -ReturnObject
    $allfiles = Invoke-Par2 -Arguments @("v", "-T4", "test.par2") -ReturnObject

    if ($onefile.ExitCode -ne $allfiles.ExitCode) {
        Exit-TestWithError "Exit code differed: -T1 gave $($onefile.ExitCode) and -T4 gave $($allfiles.ExitCode)"
    }

    # Strip the progress indicator, which is rewritten in place with `r
    $onefileBlocks = ($onefile.StdOut -replace "`r", "`n") -split "`n" | Where-Object { $_ -match "Target:" } | Sort-Object
    $allfilesBlocks = ($allfiles.StdOut -replace "`r", "`n") -split "`n" | Where-Object { $_ -match "Target:" } | Sort-Object

    if (($onefileBlocks -join "`n") -ne ($allfilesBlocks -join "`n")) {
        Write-Host "-T1:"; $onefileBlocks | ForEach-Object { Write-Host "  $_" }
        Write-Host "-T4:"; $allfilesBlocks | ForEach-Object { Write-Host "  $_" }
        Exit-TestWithError "Reading several files at once found different data"
    }

    $joined = $allfilesBlocks -join "`n"

    if ($joined -notmatch 'data1\.bin" - found') {
        Exit-TestWithError "Expected data1.bin to be found intact"
    }

    if ($joined -notmatch 'data2\.bin" - damaged\. Found 15 of 16 data blocks') {
        Exit-TestWithError "Expected 15 of 16 blocks in data2.bin"
    }

    # The repair must still work when the files were read at the same time
    $repair = Invoke-Par2 -Arguments @("r", "-q", "-T4", "test.par2") -ReturnObject
    if ($repair.ExitCode -ne 0) {
        Exit-TestWithError "Repair failed"
    }

    foreach ($n in 1..4) {
        if (-not (Compare-Files -File1 "data$n.bin" -File2 "data$n.bin.orig")) {
            Exit-TestWithError "Repaired data$n.bin does not match the original"
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
