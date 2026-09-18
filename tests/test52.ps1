#!/usr/bin/env pwsh
# Test 50: A file is found under another spelling of its name

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Write-Banner "A file is found under another spelling of its name"

    # The set records the name with the accent on the letter it belongs to,
    # and the file on disk has the accent as a character of its own, which is
    # how some file systems write it.
    $composed = "fr" + [char]0x00E8 + "nch_demo.bin"
    $decomposed = "fre" + [char]0x0300 + "nch_demo.bin"

    $builder = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt 64; $i++) {
        [void]$builder.Append("1" + $i.ToString("D63"))
    }
    [System.IO.File]::WriteAllText((Join-Path $PWD $composed), $builder.ToString())

    $create = Invoke-Par2 -Arguments @("c", "-q", "-s1024", "-c2", "test.par2", $composed) -ReturnObject
    if ($create.ExitCode -ne 0) {
        Exit-TestWithError "Could not create PAR2 files"
    }

    Move-Item $composed $decomposed
    Copy-Item $decomposed "original.bin"

    if (Test-Path $composed) {
        # This file system looks a name up under either spelling, so the file
        # is found as it is named.
        $verify = Invoke-Par2 -Arguments @("v", "-q", "test.par2") -ReturnObject
        if ($verify.ExitCode -ne 0) {
            Exit-TestWithError "The file was not found under the other spelling"
        }
    }
    else {
        # The file has the wrong name, and a repair renames it to the one the
        # set records.
        $verify = Invoke-Par2 -Arguments @("v", "test.par2") -ReturnObject
        if ($verify.ExitCode -ne 1) {
            Exit-TestWithError "Verifying the file under the other spelling gave $($verify.ExitCode), expected 1"
        }

        if (-not $verify.StdOut.Contains('"' + $decomposed + '" - is a match for "' + $composed + '"')) {
            Exit-TestWithError "The file was not matched to the name the set records"
        }

        $repair = Invoke-Par2 -Arguments @("r", "-q", "test.par2") -ReturnObject
        if ($repair.ExitCode -ne 0) {
            Exit-TestWithError "Repair failed"
        }

        if (-not (Compare-Files -File1 $composed -File2 "original.bin")) {
            Exit-TestWithError "The file was not renamed to the name the set records"
        }

        if (Test-Path $decomposed) {
            Exit-TestWithError "The file is still under the other spelling"
        }
    }

    # A file whose directory is on disk under another spelling of the
    # directory's name is found, and a repair moves it into the directory the
    # set records.
    $composeddirectory = "dir_" + $composed
    $decomposeddirectory = "dir_" + $decomposed
    New-Item -ItemType Directory -Path $composeddirectory | Out-Null
    Copy-Item "original.bin" (Join-Path $composeddirectory "plain.bin")

    $create = Invoke-Par2 -Arguments @("c", "-q", "-s1024", "-c2", "directory.par2", (Join-Path $composeddirectory "plain.bin")) -ReturnObject
    if ($create.ExitCode -ne 0) {
        Exit-TestWithError "Could not create PAR2 files"
    }

    Move-Item $composeddirectory $decomposeddirectory

    if (Test-Path $composeddirectory) {
        $verify = Invoke-Par2 -Arguments @("v", "-q", "directory.par2") -ReturnObject
        if ($verify.ExitCode -ne 0) {
            Exit-TestWithError "The file was not found under the other spelling of its directory"
        }
    }
    else {
        $verify = Invoke-Par2 -Arguments @("v", "-q", "directory.par2") -ReturnObject
        if ($verify.ExitCode -ne 1) {
            Exit-TestWithError "Verifying the file in the other spelling of its directory gave $($verify.ExitCode), expected 1"
        }

        $repair = Invoke-Par2 -Arguments @("r", "-q", "directory.par2") -ReturnObject
        if ($repair.ExitCode -ne 0) {
            Exit-TestWithError "Repair failed"
        }

        if (-not (Compare-Files -File1 (Join-Path $composeddirectory "plain.bin") -File2 "original.bin")) {
            Exit-TestWithError "The file was not moved into the directory the set records"
        }

        if (Test-Path (Join-Path $decomposeddirectory "plain.bin")) {
            Exit-TestWithError "The file is still in the other spelling of its directory"
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
