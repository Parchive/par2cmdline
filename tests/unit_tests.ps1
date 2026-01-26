#!/usr/bin/env pwsh
# Unit tests runner - runs all compiled unit test executables
# Runs all tests inline in the same console for visible output

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$script:RootDir = Split-Path -Parent $PSScriptRoot

# Find the directory containing test executables
$testDirs = @(
    (Join-Path $script:RootDir "x64\Release"),
    (Join-Path $script:RootDir "x64\Debug"),
    (Join-Path $script:RootDir "Win32\Release"),
    (Join-Path $script:RootDir "Win32\Debug"),
    (Join-Path $script:RootDir "ARM64\Release"),
    (Join-Path $script:RootDir "ARM64\Debug"),
    (Split-Path -Parent $PARBINARY)
)

$testExeDir = $null
foreach ($dir in $testDirs) {
    if (Test-Path $dir) {
        $testExeDir = $dir
        break
    }
}

if (-not $testExeDir) {
    Write-Host "ERROR: Could not find test executable directory" -ForegroundColor Red
    exit 1
}

Write-Host "Looking for unit tests in: $testExeDir" -ForegroundColor Cyan
Write-Host ""

$unitTestExes = @(
    "letype_test.exe",
    "crc_test.exe",
    "md5_test.exe",
    "diskfile_test.exe",
    "libpar2_test.exe",
    "commandline_test.exe",
    "descriptionpacket_test.exe",
    "criticalpacket_test.exe",
    "reedsolomon_test.exe",
    "galois_test.exe",
    "utf8_test.exe"
)

$passed = 0
$failed = 0
$skipped = 0
$failedTests = @()

foreach ($testExe in $unitTestExes) {
    $testPath = Join-Path $testExeDir $testExe

    if (Test-Path $testPath) {
        Write-Host "------------------------------------------------------"
        Write-Host "Running unit tests from file $testExe"
        Write-Host "------------------------------------------------------"

        $startTime = Get-Date

        try {
            # Run the executable inline using the call operator
            # This keeps output visible in the same console
            & $testPath
            $exitCode = $LASTEXITCODE

            $duration = (Get-Date) - $startTime

            if ($exitCode -eq 0) {
                Write-Host "------------------------------------------------------"
                Write-Host "PASSED: $testExe (duration: $($duration.TotalSeconds.ToString('F2'))s)" -ForegroundColor Green
                Write-Host "------------------------------------------------------"
                $passed++
            } else {
                Write-Host "------------------------------------------------------"
                Write-Host "FAILED: $testExe (exit code: $exitCode, duration: $($duration.TotalSeconds.ToString('F2'))s)" -ForegroundColor Red
                Write-Host "------------------------------------------------------"
                $failed++
                $failedTests += $testExe
            }
        }
        catch {
            $duration = (Get-Date) - $startTime
            Write-Host "------------------------------------------------------"
            Write-Host "FAILED: $testExe - $_ (duration: $($duration.TotalSeconds.ToString('F2'))s)" -ForegroundColor Red
            Write-Host "------------------------------------------------------"
            $failed++
            $failedTests += $testExe
        }

        Write-Host ""
    } else {
        Write-Host "SKIPPED: $testExe not found at $testPath" -ForegroundColor Yellow
        $skipped++
    }
}

Write-Host ""
Write-Host "======================================================"
Write-Host "Unit Test Summary"
Write-Host "======================================================"
Write-Host "Passed:  $passed" -ForegroundColor Green
Write-Host "Failed:  $failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host "Skipped: $skipped" -ForegroundColor Yellow
Write-Host ""

if ($failed -gt 0) {
    Write-Host "Failed tests:" -ForegroundColor Red
    foreach ($failedTest in $failedTests) {
        Write-Host "  - $failedTest" -ForegroundColor Red
    }
    Write-Host ""
    exit 1
}

exit 0