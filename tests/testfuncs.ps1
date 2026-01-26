# par2cmdline Windows Test Helper Functions
# This file is sourced by individual test scripts to provide common functionality

$ErrorActionPreference = "Stop"

# Determine directories
# When dot-sourced, $PSScriptRoot gives the directory of THIS file (testfuncs.ps1)
# This is more reliable than $MyInvocation.MyCommand.Path when dot-sourcing
if ($PSScriptRoot) {
    $script:TestScriptDir = $PSScriptRoot
} else {
    # Fallback for older PowerShell or unusual invocation
    $script:TestScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    if (-not $script:TestScriptDir) {
        $script:TestScriptDir = $PWD.Path
    }
}
$script:RootDir = Split-Path -Parent $script:TestScriptDir
$script:TestDataDir = $script:TestScriptDir

# Sync .NET current directory with PowerShell - needed for [System.IO.File] methods
[System.IO.Directory]::SetCurrentDirectory($PWD.Path)

# Find par2.exe binary
function Find-Par2Binary {
    # Check if PARBINARY environment variable is set
    if ($env:PARBINARY -and (Test-Path $env:PARBINARY)) {
        return $env:PARBINARY
    }

    # Check common build output locations relative to root
    $candidates = @(
        (Join-Path $script:RootDir "x64\Release\par2.exe"),
        (Join-Path $script:RootDir "x64\Debug\par2.exe"),
        (Join-Path $script:RootDir "Win32\Release\par2.exe"),
        (Join-Path $script:RootDir "Win32\Debug\par2.exe"),
        (Join-Path $script:RootDir "ARM64\Release\par2.exe"),
        (Join-Path $script:RootDir "ARM64\Debug\par2.exe"),
        (Join-Path $script:RootDir "build\par2.exe"),
        (Join-Path $script:RootDir "par2.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Could not find par2.exe. Please build the project first or set PARBINARY environment variable."
}

# Get the par2 binary path and store in environment variable for cross-script access
if (-not $env:PARBINARY -or -not (Test-Path $env:PARBINARY)) {
    $env:PARBINARY = Find-Par2Binary
}
$script:Par2Binary = $env:PARBINARY

# Extract a .tar.gz file (requires tar command available on Windows 10+)
function Expand-TarGz {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Archive,
        [Parameter(Mandatory=$true)]
        [string]$Destination
    )

    if (-not (Test-Path $Archive)) {
        throw "Archive not found: $Archive"
    }

    # Use tar command (available on Windows 10 1803+)
    $tarResult = & tar -xzf $Archive -C $Destination 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract $Archive : $tarResult"
    }
}

# Create a banner for test output
function Write-Banner {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message
    )

    $dashes = "-" * $Message.Length
    Write-Host $dashes
    Write-Host $Message
    Write-Host $dashes
}

# Run par2
# Unified function handling both console output and capturing
function Invoke-Par2 {
    param(
        [Parameter(Mandatory=$true)]
        [string[]]$Arguments,
        
        [Parameter(Mandatory=$false)]
        [switch]$ReturnObject
    )

    # Get the par2 binary path from environment variable
    $par2Path = $env:PARBINARY
    if (-not $par2Path) {
        throw "PARBINARY environment variable not set"
    }

    $tempOut = [System.IO.Path]::GetTempFileName()
    $tempErr = [System.IO.Path]::GetTempFileName()

    try {
        # Use Start-Process with explicit WorkingDirectory for output capture
        $process = Start-Process -FilePath $par2Path -ArgumentList $Arguments -NoNewWindow -Wait -PassThru -WorkingDirectory $PWD.Path -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr
        
        $stdOutContent = Get-Content $tempOut -Raw
        if ($null -eq $stdOutContent) { $stdOutContent = "" }
        
        $stdErrContent = Get-Content $tempErr -Raw
        if ($null -eq $stdErrContent) { $stdErrContent = "" }

        if ($ReturnObject) {
            return [pscustomobject]@{
                ExitCode = $process.ExitCode
                StdOut = $stdOutContent
                StdErr = $stdErrContent
            }
        } else {
            # Display output if not returning object
            if ($stdOutContent) {
                Write-Host $stdOutContent
            }
            if ($stdErrContent) {
                Write-Host $stdErrContent -ForegroundColor Yellow
            }
            return $process.ExitCode
        }
    }
    finally {
        if (Test-Path $tempOut) { Remove-Item $tempOut -Force -ErrorAction SilentlyContinue }
        if (Test-Path $tempErr) { Remove-Item $tempErr -Force -ErrorAction SilentlyContinue }
    }
}

# Compare two files by hash
function Compare-Files {
    param(
        [Parameter(Mandatory=$true)]
        [string]$File1,
        [Parameter(Mandatory=$true)]
        [string]$File2
    )

    if (-not (Test-Path $File1)) { return $false }
    if (-not (Test-Path $File2)) { return $false }

    $hash1 = Get-FileHash -Path $File1 -Algorithm SHA256
    $hash2 = Get-FileHash -Path $File2 -Algorithm SHA256

    return $hash1.Hash -eq $hash2.Hash
}

# Create random data file
function New-RandomFile {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Path,
        [Parameter(Mandatory=$true)]
        [int]$SizeBytes
    )

    $bytes = New-Object byte[] $SizeBytes
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    [System.IO.File]::WriteAllBytes($Path, $bytes)
    $rng.Dispose()
}

# Setup test environment - creates run directory and changes to it
function Initialize-Test {
    param(
        [Parameter(Mandatory=$true)]
        [string]$TestName
    )

    # Use the test script directory as root, not the current working directory
    $script:TestRoot = $script:TestScriptDir
    $script:RunDir = Join-Path $script:TestRoot "run$TestName"

    # Cleanup any previous run
    if (Test-Path $script:RunDir) {
        Remove-Item -Recurse -Force $script:RunDir
    }

    New-Item -ItemType Directory -Path $script:RunDir -Force | Out-Null
    Set-Location $script:RunDir
    
    # Sync .NET current directory with PowerShell
    [System.IO.Directory]::SetCurrentDirectory($PWD.Path)
}

# Cleanup test environment
function Complete-Test {
    Set-Location $script:TestRoot
    if (Test-Path $script:RunDir) {
        Remove-Item -Recurse -Force $script:RunDir -ErrorAction SilentlyContinue
    }
}

# Exit with error
function Exit-TestWithError {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message
    )

    Write-Host "ERROR: $Message" -ForegroundColor Red
    Complete-Test
    exit 1
}

# Export commonly needed variables
$script:TESTDATA = $script:TestDataDir
$script:PARBINARY = $env:PARBINARY