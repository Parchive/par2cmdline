#!/usr/bin/env pwsh
# par2cmdline Windows Test Runner
# PowerShell script to run tests similar to 'make check' on Unix systems
# This runner discovers and executes individual test files (test*.ps1)
# All tests run inline in the same console for visible output

param(
    [string]$Par2Binary = "",
    [switch]$Verbose,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"
$script:TestsPassed = 0
$script:TestsFailed = 0
$script:TestsSkipped = 0
$script:FailedTests = @()

# Get script and project directories
$script:ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:RootDir = Split-Path -Parent $script:ScriptDir

# Setup transcript logging if requested - captures ALL output including from child scripts
$script:TranscriptStarted = $false
if ($LogFile -ne "") {
    try {
        # Resolve to absolute path
        if (-not [System.IO.Path]::IsPathRooted($LogFile)) {
            $LogFile = Join-Path $PWD.Path $LogFile
        }
        Start-Transcript -Path $LogFile -Force | Out-Null
        $script:TranscriptStarted = $true
    }
    catch {
        Write-Host "WARNING: Could not start transcript logging: $_" -ForegroundColor Yellow
    }
}

Write-Host "========================================" -ForegroundColor Magenta
Write-Host "par2cmdline Windows Test Suite" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
if ($script:TranscriptStarted) {
    Write-Host "Logging to: $LogFile" -ForegroundColor Cyan
}
Write-Host ""

# Set PARBINARY environment variable if specified
if ($Par2Binary -ne "") {
    if (Test-Path $Par2Binary) {
        $env:PARBINARY = (Resolve-Path $Par2Binary).Path
        Write-Host "Using par2 binary: $env:PARBINARY" -ForegroundColor Cyan
    } else {
        Write-Host "ERROR: Specified par2 binary not found: $Par2Binary" -ForegroundColor Red
        if ($script:TranscriptStarted) { Stop-Transcript | Out-Null }
        exit 1
    }
}

Write-Host ""

# Function to run a single test script inline
# Uses Push-Location/Pop-Location like a subshell - test can change directory freely
# and we always return to the original location
function Invoke-SingleTest {
    param(
        [string]$TestScript
    )

    $testName = [System.IO.Path]::GetFileNameWithoutExtension($TestScript)

    Write-Host "----------------------------------------" -ForegroundColor Cyan
    Write-Host "Running: $testName" -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Cyan

    $startTime = Get-Date

    # Push current location - like entering a subshell
    Push-Location

    try {
        # Run the test script inline using the call operator
        # This keeps everything in the same console window and transcript
        # The test can use Set-Location freely - we'll pop back after
        & $TestScript
        $exitCode = $LASTEXITCODE

        $duration = (Get-Date) - $startTime

        if ($exitCode -eq 0) {
            Write-Host "PASSED: $testName (duration: $($duration.TotalSeconds.ToString('F2'))s)" -ForegroundColor Green
            $script:TestsPassed++
            return $true
        } else {
            Write-Host "FAILED: $testName (exit code: $exitCode, duration: $($duration.TotalSeconds.ToString('F2'))s)" -ForegroundColor Red
            $script:TestsFailed++
            $script:FailedTests += $testName
            return $false
        }
    }
    catch {
        $duration = (Get-Date) - $startTime
        Write-Host "FAILED: $testName - $_ (duration: $($duration.TotalSeconds.ToString('F2'))s)" -ForegroundColor Red
        $script:TestsFailed++
        $script:FailedTests += $testName
        return $false
    }
    finally {
        # Pop back to original location - like exiting a subshell
        Pop-Location
    }
}

# ============================================================================
# Run Unit Tests
# ============================================================================

$unitTestScript = Join-Path $script:ScriptDir "unit_tests.ps1"
if (Test-Path $unitTestScript) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Running Unit Tests" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # Push location like a subshell
    Push-Location

    $startTime = Get-Date

    try {
        # Run unit tests inline
        & $unitTestScript
        $exitCode = $LASTEXITCODE

        $duration = (Get-Date) - $startTime

        if ($exitCode -eq 0) {
            Write-Host "PASSED: unit_tests (duration: $($duration.TotalSeconds.ToString('F2'))s)" -ForegroundColor Green
            $script:TestsPassed++
        } else {
            Write-Host "FAILED: unit_tests (exit code: $exitCode, duration: $($duration.TotalSeconds.ToString('F2'))s)" -ForegroundColor Red
            $script:TestsFailed++
            $script:FailedTests += "unit_tests"
        }
    }
    catch {
        $duration = (Get-Date) - $startTime
        Write-Host "FAILED: unit_tests - $_ (duration: $($duration.TotalSeconds.ToString('F2'))s)" -ForegroundColor Red
        $script:TestsFailed++
        $script:FailedTests += "unit_tests"
    }
    finally {
        # Pop back to original location
        Pop-Location
    }

    Write-Host ""
}

# ============================================================================
# Discover and Run Integration Tests
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running Integration Tests" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Find all test*.ps1 files (excluding testfuncs.ps1, unit_tests.ps1, run_tests.ps1, build_unit_tests.ps1)
$testScripts = Get-ChildItem -Path $script:ScriptDir -Filter "test*.ps1" |
    Where-Object {
        $_.Name -ne "testfuncs.ps1" -and
        $_.Name -notlike "*_test*.ps1" -and
        $_.Name -ne "run_tests.ps1"
    } |
    Sort-Object {
        # Sort numerically by test number
        if ($_.BaseName -match '^test(\d+)') {
            [int]$matches[1]
        } elseif ($_.BaseName -match '^test(\d+)(\w+)$') {
            # Handle tests like test5rk - sort after test5
            [int]$matches[1] + 0.5
        } else {
            999
        }
    }

$totalTests = $testScripts.Count
$currentTest = 0

foreach ($testScript in $testScripts) {
    $testName = $testScript.BaseName
    $currentTest++

    Write-Host "[$currentTest/$totalTests] " -ForegroundColor White -NoNewline
    $result = Invoke-SingleTest -TestScript $testScript.FullName

    Write-Host ""
}

# ============================================================================
# Summary
# ============================================================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "Test Results Summary" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ""
Write-Host "Passed:  $script:TestsPassed" -ForegroundColor Green
Write-Host "Failed:  $script:TestsFailed" -ForegroundColor $(if ($script:TestsFailed -gt 0) { "Red" } else { "Green" })
Write-Host "Skipped: $script:TestsSkipped" -ForegroundColor Yellow
Write-Host ""

if ($script:TestsFailed -gt 0) {
    Write-Host "Failed tests:" -ForegroundColor Red
    foreach ($failedTest in $script:FailedTests) {
        Write-Host "  - $failedTest" -ForegroundColor Red
    }
    Write-Host ""
}

if ($script:TranscriptStarted) {
    Write-Host "Full log saved to: $LogFile" -ForegroundColor Cyan
    Stop-Transcript | Out-Null
}

if ($script:TestsFailed -gt 0) {
    exit 1
}

exit 0