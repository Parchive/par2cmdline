#!/usr/bin/env pwsh
# Test 57: Purge after swapped file names

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

    foreach ($f in @("test-1.data", "test-2.data")) {
        New-RandomFile -Path $f -SizeBytes 204800
    }
    Copy-Item "test-1.data" "test-1.data.orig"
    Copy-Item "test-2.data" "test-2.data.orig"

    $exitCode = Invoke-Par2 -Arguments @("c", "-b40", "-c40", "testdata.par2", "test-1.data", "test-2.data")
    if ($exitCode -ne 0) {
        Exit-TestWithError "creating repair files using PAR 2.0 failed"
    }

    Write-Banner "purge keeps a complete file found under another file's name"

    Move-Item "test-1.data" "newname"
    Move-Item "test-2.data" "test-1.data"

    $result = Invoke-Par2 -Arguments @("r", "-p", "testdata.par2", "newname") -ReturnObject
    Write-Host $result.StdOut
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "repair using PAR 2.0 failed"
    }
    if (-not (Compare-Files "test-1.data" "test-1.data.orig")) {
        Exit-TestWithError "test-1.data does not match the original"
    }
    if (-not (Compare-Files "test-2.data" "test-2.data.orig")) {
        Exit-TestWithError "test-2.data was purged"
    }

    Write-Banner "purge removes a damaged file found under another file's name"

    $exitCode = Invoke-Par2 -Arguments @("c", "-b40", "-c40", "testdata.par2", "test-1.data", "test-2.data")
    if ($exitCode -ne 0) {
        Exit-TestWithError "creating repair files using PAR 2.0 failed"
    }

    Move-Item "test-1.data" "newname"
    Move-Item "test-2.data" "test-1.data"
    Set-Bytes -Path "test-1.data" -Offset 100000 -Bytes ([byte[]]@(0xde, 0xad, 0xbe, 0xef))

    $result = Invoke-Par2 -Arguments @("r", "-p", "testdata.par2", "newname") -ReturnObject
    Write-Host $result.StdOut
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "repair using PAR 2.0 failed"
    }
    if (-not (Compare-Files "test-1.data" "test-1.data.orig")) {
        Exit-TestWithError "test-1.data does not match the original"
    }
    if (-not (Compare-Files "test-2.data" "test-2.data.orig")) {
        Exit-TestWithError "test-2.data does not match the original"
    }
    if (Test-Path "test-1.data.1") {
        Exit-TestWithError "the damaged copy was not purged"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
