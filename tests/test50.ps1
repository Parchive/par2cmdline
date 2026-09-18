#!/usr/bin/env pwsh
# Test 50: What the tool prints is what was recorded

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

$actual = New-Object System.Text.StringBuilder

# Record what a run prints, so that a change to it has to be made deliberately.
# Files are read one at a time, so they are reported in the order the set
# records them. The progress indicator rewrites its line with `r, so each line
# is reduced to what was left on it, which is what a console shows.
function Add-Run {
    param(
        [Parameter(Mandatory=$true)]
        [string[]]$Arguments
    )

    $result = Invoke-Par2 -Arguments $Arguments -ReturnObject -RawOutput

    [void]$actual.Append('$ par2 ' + ($Arguments -join ' ') + "`n")
    [void]$actual.Append("exit $($result.ExitCode)`n")

    $text = $result.StdOut -replace "`r`n", "`n"
    if ($text.EndsWith("`n")) {
        $text = $text.Substring(0, $text.Length - 1)
    }
    if ($text -ne "") {
        foreach ($line in ($text -split "`n")) {
            [void]$actual.Append(($line -replace '.*\r', '') + "`n")
        }
    }

    [void]$actual.Append("`n")
}

try {
    Initialize-Test -TestName $testname

    Write-Banner "What the tool prints is what was recorded"

    # Three data files of four blocks of 1024 bytes. Each file holds different
    # data so that a block cannot be matched against the wrong file.
    foreach ($n in 1..3) {
        $builder = New-Object System.Text.StringBuilder
        for ($i = 0; $i -lt 64; $i++) {
            [void]$builder.Append($n.ToString() + $i.ToString("D63"))
        }
        [System.IO.File]::WriteAllText((Join-Path $PWD "data$n.bin"), $builder.ToString())
    }

    Add-Run @("c", "-T1", "-s1024", "-c4", "-n1", "test.par2", "data1.bin", "data2.bin", "data3.bin")
    Add-Run @("v", "-T1", "test.par2")

    # One block of data2.bin is corrupt, which the recovery blocks can replace.
    $bytes = [System.IO.File]::ReadAllBytes((Join-Path $PWD "data2.bin"))
    for ($i = 0; $i -lt 16; $i++) {
        $bytes[2048 + $i] = [byte][char]'X'
    }
    [System.IO.File]::WriteAllBytes((Join-Path $PWD "data2.bin"), $bytes)

    Add-Run @("v", "-T1", "test.par2")
    Add-Run @("r", "-T1", "test.par2")

    # More blocks are lost than there are recovery blocks to replace them.
    Remove-Item "data3.bin" -Force
    $bytes = [System.IO.File]::ReadAllBytes((Join-Path $PWD "data1.bin"))
    foreach ($off in 0, 1024, 2048, 3072) {
        for ($i = 0; $i -lt 16; $i++) {
            $bytes[$off + $i] = [byte][char]'X'
        }
    }
    [System.IO.File]::WriteAllBytes((Join-Path $PWD "data1.bin"), $bytes)

    Add-Run @("v", "-T1", "test.par2")

    $recorded = [System.IO.File]::ReadAllText((Join-Path $TESTDATA "$testname.expected")) -replace "`r`n", "`n"

    if ($actual.ToString() -ne $recorded) {
        Compare-Object -ReferenceObject ($recorded -split "`n") `
                       -DifferenceObject ($actual.ToString() -split "`n") |
            ForEach-Object { Write-Host "$($_.SideIndicator) $($_.InputObject)" }
        Exit-TestWithError "The tool printed something other than what was recorded"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
