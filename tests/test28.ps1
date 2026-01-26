#!/usr/bin/env pwsh
# Test 28: Ensuring silent noise level (-qq flag)

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname

    Expand-TarGz -Archive (Join-Path $TESTDATA "flatdata.tar.gz") -Destination "."

    Write-Banner "Ensuring silent noise level"

    $dataFiles = @("test-0.data", "test-1.data", "test-2.data", "test-3.data", "test-4.data", "test-5.data", "test-6.data", "test-7.data", "test-8.data", "test-9.data")

    # Create with -qq should produce no stdout
    $result = Invoke-Par2 -Arguments (@("c", "-a", "newtest", "-qq") + $dataFiles) -ReturnObject
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "create failed"
    }
    if ($result.StdOut -and $result.StdOut.Trim().Length -gt 0) {
        Exit-TestWithError "create with -qq produced output to stdout"
    }

    # Verify with -qq should produce no stdout
    $result = Invoke-Par2 -Arguments (@("v", "-qq", "newtest") + $dataFiles) -ReturnObject
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "verify failed"
    }
    if ($result.StdOut -and $result.StdOut.Trim().Length -gt 0) {
        Exit-TestWithError "verify with -qq produced output to stdout"
    }

    # Repair with -qq should produce no stdout
    $result = Invoke-Par2 -Arguments (@("r", "-qq", "newtest") + $dataFiles) -ReturnObject
    if ($result.ExitCode -ne 0) {
        Exit-TestWithError "repair failed"
    }
    if ($result.StdOut -and $result.StdOut.Trim().Length -gt 0) {
        Exit-TestWithError "repair with -qq produced output to stdout"
    }

    # Second create should fail (files already exist) but still be silent on stdout
    $result = Invoke-Par2 -Arguments (@("c", "-a", "newtest", "-qq") + $dataFiles) -ReturnObject
    if ($result.ExitCode -eq 0) {
        Exit-TestWithError "second create succeeded but shouldn't have"
    }
    if ($result.StdOut -and $result.StdOut.Trim().Length -gt 0) {
        Exit-TestWithError "second create with -qq produced output to stdout"
    }
    if (-not $result.StdErr -or $result.StdErr.Trim().Length -eq 0) {
        Exit-TestWithError "second create with -qq did not produce output to stderr"
    }

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}