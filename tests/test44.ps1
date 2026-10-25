#!/usr/bin/env pwsh
# Test 44: par2 create works with more than 2000 input files

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname
    $runDir = $PWD.Path

    Write-Banner "par2 create works with more than 2000 input files"

    # Generate 2001 non-empty data files (each 512 bytes).
    # The old hard-coded blockcount=2000 default would fail here because the
    # number of source files exceeded the block count.
    $datadir = Join-Path $runDir "manyfiles"
    New-Item -ItemType Directory -Path $datadir -Force | Out-Null

    for ($i = 0; $i -lt 2001; $i++) {
        $name = "file_{0:D4}.data" -f $i
        [System.IO.File]::WriteAllText(
            (Join-Path $datadir $name),
            $i.ToString().PadLeft(512, '0'))
    }

    $create = Invoke-Par2 -Arguments @(
        "c",
        "-B$datadir",
        (Join-Path $datadir "test.par2"),
        (Join-Path $datadir "*.data")) -ReturnObject

    if ($create.ExitCode -ne 0) {
        Write-Host $create.StdOut
        Write-Host $create.StdErr -ForegroundColor Yellow
        Exit-TestWithError "create with >2000 files failed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
