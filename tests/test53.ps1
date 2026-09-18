#!/usr/bin/env pwsh
# Test 53: A file is found when the set records its name in a code page

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "A file is found when the set records its name in a code page"

    # The set was written by a machine using a code page, so it records the
    # name in Windows-1252 while the file on disk is named in UTF-8.
    Expand-TarGz -Archive (Join-Path $TESTDATA "codepage-name.tar.gz") -Destination "."

    $name = "data_" + [char]0x00E8 + ".bin"
    Copy-Item $name "original.bin"

    $verify = Invoke-Par2 -Arguments @("v", "-q", "recovery.par2") -ReturnObject
    if ($verify.ExitCode -ne 0) {
        Exit-TestWithError "The file was not found under its UTF-8 name"
    }

    # The repair has to write to the file which is there, not to the name the
    # set records.
    $bytes = [System.IO.File]::ReadAllBytes((Join-Path $PWD $name))
    for ($i = 0; $i -lt 16; $i++) {
        $bytes[2048 + $i] = [byte][char]'X'
    }
    [System.IO.File]::WriteAllBytes((Join-Path $PWD $name), $bytes)

    $repair = Invoke-Par2 -Arguments @("r", "-q", "recovery.par2") -ReturnObject
    if ($repair.ExitCode -ne 0) {
        Exit-TestWithError "Repair failed"
    }

    if (-not (Compare-Files -File1 $name -File2 "original.bin")) {
        Exit-TestWithError "The repaired file does not match the original"
    }

    # A file which is missing is written under its name in UTF-8.
    Remove-Item $name

    $repair = Invoke-Par2 -Arguments @("r", "-q", "recovery.par2") -ReturnObject
    if ($repair.ExitCode -ne 0) {
        Exit-TestWithError "Repair of the missing file failed"
    }

    if (-not (Compare-Files -File1 $name -File2 "original.bin")) {
        Exit-TestWithError "The missing file was not written under its UTF-8 name"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
