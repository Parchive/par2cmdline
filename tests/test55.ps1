#!/usr/bin/env pwsh
# Test 55: Repair damaged files in place (-i)

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

function Set-Bytes {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][int]$Offset,
        [Parameter(Mandatory=$true)][byte[]]$Bytes
    )

    $stream = [System.IO.File]::Open($Path, 'Open', 'Write')
    try {
        $stream.Seek($Offset, 'Begin') | Out-Null
        $stream.Write($Bytes, 0, $Bytes.Length)
    }
    finally {
        $stream.Dispose()
    }
}

try {
    Initialize-Test -TestName $testname

    # Data which does not repeat, so that every block is found at its own offset
    foreach ($f in @("test-1.data", "test-2.data", "test-3.data")) {
        New-RandomFile -Path $f -SizeBytes 204800
    }
    Copy-Item "test-1.data" "test-1.data.orig"
    Copy-Item "test-2.data" "test-2.data.orig"

    # 24 recovery blocks rebuild a whole file of 20 blocks
    $exitCode = Invoke-Par2 -Arguments @("c", "-b60", "-c24", "testdata.par2", "test-1.data", "test-2.data", "test-3.data")
    if ($exitCode -ne 0) {
        Exit-TestWithError "creating repair files using PAR 2.0 failed"
    }

    Write-Banner "repair a damaged file in place"

    Set-Bytes -Path "test-1.data" -Offset 100000 -Bytes ([byte[]]@(0xde, 0xad, 0xbe, 0xef))

    $result = Invoke-Par2 -Arguments @("r", "-i", "testdata.par2") -ReturnObject
    Write-Host $result.StdOut
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "in-place repair using PAR 2.0 failed"
    }
    if ($result.StdOut -notmatch "Rewrote .* in place") {
        Exit-TestWithError "the file was not repaired in place"
    }
    if (Test-Path "test-1.data.1") {
        Exit-TestWithError "in-place repair created a backup copy"
    }
    if (-not (Compare-Files "test-1.data" "test-1.data.orig")) {
        Exit-TestWithError "the in-place repaired file does not match the original"
    }

    Write-Banner "fall back to a new copy when the file is the wrong size"

    $bytes = [System.IO.File]::ReadAllBytes("test-1.data.orig")
    [System.IO.File]::WriteAllBytes("test-1.data", $bytes[0..($bytes.Length - 2)])

    $result = Invoke-Par2 -Arguments @("r", "-i", "testdata.par2") -ReturnObject
    Write-Host $result.StdOut
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "fallback repair using PAR 2.0 failed"
    }
    if ($result.StdOut -notmatch "Cannot repair") {
        Exit-TestWithError "no reason was given for not repairing in place"
    }
    if (-not (Test-Path "test-1.data.1")) {
        Exit-TestWithError "fallback repair did not create a backup copy"
    }
    if (-not (Compare-Files "test-1.data" "test-1.data.orig")) {
        Exit-TestWithError "the repaired file does not match the original"
    }
    Remove-Item "test-1.data.1" -Force

    Write-Banner "do not repair in place a file which holds another file's data"

    # test-1.data holds the only copy of test-2.data
    Move-Item "test-2.data" "test-1.data" -Force

    $result = Invoke-Par2 -Arguments @("r", "-i", "testdata.par2") -ReturnObject
    Write-Host $result.StdOut
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "repair using PAR 2.0 failed"
    }
    if ($result.StdOut -notmatch "Cannot repair") {
        Exit-TestWithError "no reason was given for not repairing in place"
    }
    if (-not (Compare-Files "test-1.data" "test-1.data.orig")) {
        Exit-TestWithError "test-1.data does not match the original"
    }
    if (-not (Compare-Files "test-2.data" "test-2.data.orig")) {
        Exit-TestWithError "test-2.data was destroyed"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
