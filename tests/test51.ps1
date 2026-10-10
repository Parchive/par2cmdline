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

    # Every file named by its long path, in a directory whose name is not
    # ASCII, with the directory's files given as a wildcard.
    $directory = Join-Path (Get-Location).Path ([string][char]0x0119 + "_" + [char]0x4F60 + [char]0x597D)
    New-Item -ItemType Directory -Path $directory | Out-Null
    Copy-Item -Path "unicode*.par2" -Destination $directory
    Copy-Item "original.bin" (Join-Path $directory "data_ascii.bin")
    $long = "\\?\" + $directory

    $repair = Invoke-Par2 -Arguments @("r", "-B", $long, "$long\unicode.par2", "$long\*") -ReturnObject
    if ($repair.ExitCode -ne 0) {
        Exit-TestWithError "Repair through long paths gave $($repair.ExitCode), expected 0"
    }

    if (-not (Compare-Files -File1 (Join-Path $directory $unicode) -File2 "original.bin")) {
        Exit-TestWithError "The file was not renamed to its unicode name through long paths"
    }

    # A set made in that directory through long paths verifies the same way.
    $create = Invoke-Par2 -Arguments @("c", "-q", "-B", $long, "$long\made.par2", "$long\$unicode") -ReturnObject
    if ($create.ExitCode -ne 0) {
        Exit-TestWithError "Creating through long paths gave $($create.ExitCode), expected 0"
    }

    $verify = Invoke-Par2 -Arguments @("v", "-q", "-B", $long, "$long\made.par2") -ReturnObject
    if ($verify.ExitCode -ne 0) {
        Exit-TestWithError "Verifying through long paths gave $($verify.ExitCode), expected 0"
    }

    # A damaged copy under the name in the description packet is renamed out
    # of the way, and purged, when the repair rebuilds the file under its
    # unicode name.
    Remove-Item $unicode
    Copy-Item "original.bin" "data_ascii.bin"
    $bytes = [System.IO.File]::ReadAllBytes((Join-Path $PWD "data_ascii.bin"))
    for ($i = 0; $i -lt 16; $i++) {
        $bytes[2048 + $i] = [byte][char]'X'
    }
    [System.IO.File]::WriteAllBytes((Join-Path $PWD "data_ascii.bin"), $bytes)

    $repair = Invoke-Par2 -Arguments @("r", "-q", "-p", "unicode.par2") -ReturnObject
    if ($repair.ExitCode -ne 0) {
        Exit-TestWithError "Repair of the damaged copy failed"
    }

    if (-not (Compare-Files -File1 $unicode -File2 "original.bin")) {
        Exit-TestWithError "The file was not rebuilt under its unicode name"
    }

    if ((Test-Path "data_ascii.bin") -or (Test-Path "data_ascii.bin.1")) {
        Exit-TestWithError "The damaged copy was left behind"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
