#!/usr/bin/env pwsh
# Test 46: Verify -L (follow symlinks) handling

$ErrorActionPreference = "Stop"

# Source common test functions
. (Join-Path $PSScriptRoot "testfuncs.ps1")

$testname = [System.IO.Path]::GetFileNameWithoutExtension($MyInvocation.MyCommand.Name)

try {
    Initialize-Test -TestName $testname
    $runDir = $PWD.Path

    Write-Banner "Verify -L (follow symlinks) handling"

    # generate test data and links
    [System.IO.File]::WriteAllText((Join-Path $runDir "test_file"), '01234567890abcdef')
    New-Symlink -Path "test_link" -Target "test_file"
    New-Item -ItemType Directory -Path "test_dir" -Force | Out-Null
    New-Symlink -Path "test_dir\test_link" -Target (Join-Path $runDir "test_file")
    New-Symlink -Path "dir_link" -Target (Join-Path $runDir "test_dir")
    New-Symlink -Path "dead_link" -Target "NOT_EXISTING"

    # without -L, links are not followed for non-primary files
    if ((Invoke-Par2 -Arguments @("c", "-r10", "nofollow", "test_link") -ReturnObject).ExitCode -eq 0) {
        Exit-TestWithError "followed link without -L"
    }

    # test creation with symlink following
    Write-Host "*** Create with -L ***"
    if ((Invoke-Par2 -Arguments @("c", "-L", "-r10", "follow_links", "test_link") -ReturnObject).ExitCode -ne 0) {
        Exit-TestWithError "creating with -L failed"
    }

    # the linked data verifies and repairs
    Write-Host "*** Verify good file link ***"
    if ((Invoke-Par2 -Arguments @("v", "-q", "follow_links") -ReturnObject).ExitCode -ne 0) {
        Exit-TestWithError "verifying good link data failed"
    }

    Copy-Item "test_file" "test_file.bak"
    [System.IO.File]::AppendAllText((Join-Path $runDir "test_file"), '!')

    Write-Host "*** Verify damaged file link ***"
    if ((Invoke-Par2 -Arguments @("v", "-q", "follow_links") -ReturnObject).ExitCode -eq 0) {
        Exit-TestWithError "verifying damaged link data succeeded"
    }

    Write-Host "*** Repair ***"
    if ((Invoke-Par2 -Arguments @("r", "-q", "follow_links") -ReturnObject).ExitCode -ne 0) {
        Exit-TestWithError "repairing linked data failed"
    }

    if (-not (Compare-Files -File1 "test_link" -File2 "test_file.bak")) {
        Exit-TestWithError "comparison after repairing linked data failed"
    }

    # with -R -L, links to files inside directories are followed
    Write-Host "*** Create recursive with -L ***"
    if ((Invoke-Par2 -Arguments @("c", "-R", "-L", "-r10", "dirrec", "test_dir") -ReturnObject).ExitCode -ne 0) {
        Exit-TestWithError "recursive create with -L failed"
    }

    if ((Invoke-Par2 -Arguments @("v", "-q", "dirrec") -ReturnObject).ExitCode -ne 0) {
        Exit-TestWithError "verifying recursive link data failed"
    }

    # linked directories are not followed, even with -R -L
    if ((Invoke-Par2 -Arguments @("c", "-R", "-L", "-r10", "dirlink", "dir_link") -ReturnObject).ExitCode -eq 0) {
        Exit-TestWithError "followed linked directory"
    }

    # dead links fail, even with -L
    if ((Invoke-Par2 -Arguments @("c", "-L", "-r10", "dead", "dead_link") -ReturnObject).ExitCode -eq 0) {
        Exit-TestWithError "calculating dead link data succeeded"
    }

    Write-Host ""
    Write-Host "Link tests succeeded!"

    Complete-Test
    exit 0
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Complete-Test
    exit 1
}
