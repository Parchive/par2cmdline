#!/usr/bin/env pwsh
# Test 51: A file is named by the unicode filename packet the set gives it

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "A file is named by the unicode filename packet the set gives it"

    Expand-TarGz -Archive (Join-Path $TESTDATA "unicode-name.tar.gz") -Destination "."

    # unicode.par2 names the file data_ascii.bin in its description packet and
    # this in its unicode filename packet, which takes its place.
    $unicode = "data_" + [char]0x00E9 + "_" + [char]::ConvertFromUtf32(0x1F600) + ".bin"
    Copy-Item "data_ascii.bin" "original.bin"

    Move-Item "data_ascii.bin" $unicode

    $verify = Invoke-Par2 -Arguments @("v", "-q", "unicode.par2") -ReturnObject
    if ($verify.ExitCode -ne 0) {
        Exit-TestWithError "The file was not found under its unicode name"
    }

    # A file under the name in the description packet has the wrong name, and a
    # repair renames it.
    Move-Item $unicode "data_ascii.bin"

    $verify = Invoke-Par2 -Arguments @("v", "unicode.par2") -ReturnObject
    if ($verify.ExitCode -ne 1) {
        Exit-TestWithError "Verifying the file under its description packet's name gave $($verify.ExitCode), expected 1"
    }

    if (-not $verify.StdOut.Contains('"data_ascii.bin" - is a match for "' + $unicode + '"')) {
        Exit-TestWithError "The file was not matched to its unicode name"
    }

    $repair = Invoke-Par2 -Arguments @("r", "-q", "unicode.par2") -ReturnObject
    if ($repair.ExitCode -ne 0) {
        Exit-TestWithError "Repair failed"
    }

    if (-not (Compare-Files -File1 $unicode -File2 "original.bin")) {
        Exit-TestWithError "The file was not renamed to its unicode name"
    }

    if (Test-Path "data_ascii.bin") {
        Exit-TestWithError "The file is still under its description packet's name"
    }

    # bad.par2 has a unicode filename packet which is not UTF-16, so the name
    # in its description packet is used.
    $bad = Invoke-Par2 -Arguments @("v", "bad.par2") -ReturnObject
    if ($bad.ExitCode -ne 0) {
        Exit-TestWithError "The file was not found under its description packet's name"
    }

    if (-not ($bad.StdOut + $bad.StdErr).Contains("does not hold UTF-16")) {
        Exit-TestWithError "The packet which is not UTF-16 was not reported"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
