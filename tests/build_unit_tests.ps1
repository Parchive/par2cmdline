# par2cmdline Windows Unit Test Builder
# PowerShell script to compile unit tests using Visual C++ compiler (cl.exe)

param(
    [string]$Configuration = "Release",
    [string]$Platform = "x64",
    [switch]$Clean,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Get script and project directories
$script:ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:RootDir = Split-Path -Parent $script:ScriptDir
$script:SrcDir = Join-Path $script:RootDir "src"
$script:OutputDir = Join-Path $script:RootDir "$Platform\$Configuration"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "par2cmdline Unit Test Builder" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuration: $Configuration"
Write-Host "Platform: $Platform"
Write-Host "Output Directory: $script:OutputDir"
Write-Host ""

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning previous unit test builds..." -ForegroundColor Yellow
    $testExes = @(
        "letype_test",
        "crc_test",
        "md5_test",
        "diskfile_test",
        "libpar2_test",
        "commandline_test",
        "descriptionpacket_test",
        "criticalpacket_test",
        "reedsolomon_test",
        "galois_test",
        "utf8_test"
    )
    $ObjDir = Join-Path $script:RootDir "tests\$Platform\$Configuration"
    foreach ($exe in $testExes) {
        $exePath = Join-Path $script:OutputDir "$exe.exe"
        if (Test-Path $exePath) {
            Remove-Item $exePath -Force
            Write-Host "  Removed: $exe.exe"
        }
        # Clean object files
        $objPath = Join-Path $ObjDir $exe
        if (Test-Path $objPath) {
            Remove-Item $objPath -Force -Recurse
            Write-Host "  Removed: $exe\*"
        }
    }
    Write-Host ""
}

# Find Visual Studio and set up environment
function Find-VSEnvironment {
    # Try to find vswhere
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    
    if (-not (Test-Path $vsWhere)) {
        throw "Could not find vswhere.exe. Please ensure Visual Studio is installed."
    }
    
    # Find latest VS installation
    $vsPath = & $vsWhere -latest -property installationPath
    if (-not $vsPath) {
        throw "Could not find Visual Studio installation."
    }
    
    # Find vcvarsall.bat
    $vcvarsall = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
    if (-not (Test-Path $vcvarsall)) {
        throw "Could not find vcvarsall.bat at: $vcvarsall"
    }
    
    return $vcvarsall
}

function Invoke-VCCommand {
    param(
        [string]$VcVarsAll,
        [string]$Platform,
        [string]$Command
    )
    
    # Map platform to vcvarsall argument
    $vcArch = switch ($Platform) {
        "x64" { "x64" }
        "Win32" { "x86" }
        "ARM64" { "arm64" }
        default { "x64" }
    }
    
    # Create a batch file that sets up environment and runs command
    $batchContent = @"
@echo off
call "$VcVarsAll" $vcArch >nul 2>&1
$Command
"@
    
    $tempBatch = Join-Path $env:TEMP "par2_build_$(Get-Random).bat"
    Set-Content -Path $tempBatch -Value $batchContent -Encoding ASCII
    
    try {
        $output = & cmd.exe /c $tempBatch 2>&1
        $exitCode = $LASTEXITCODE
        return @{
            Output = $output
            ExitCode = $exitCode
        }
    }
    finally {
        Remove-Item $tempBatch -Force -ErrorAction SilentlyContinue
    }
}

# Find VS environment
Write-Host "Finding Visual Studio environment..." -ForegroundColor Cyan
try {
    $vcvarsall = Find-VSEnvironment
    Write-Host "  Found: $vcvarsall" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Build command
$clCommand = "msbuild.exe -m -property:Configuration=UnitTests-$Configuration -property:Platform=$Platform $script:RootDir\par2cmdline.sln"

if ($Verbose) {
    Write-Host "  Command: $clCommand" -ForegroundColor Gray
}

$result = Invoke-VCCommand -VcVarsAll $vcvarsall -Platform $Platform -Command $clCommand

if ($result.ExitCode -ne 0) {
    if ($Verbose -or $true) {
        $result.Output | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    }
    Write-Host "Some unit tests failed to build. Check the output above for details." -ForegroundColor Yellow
    exit 1
}

Write-Host "All unit tests built successfully!" -ForegroundColor Green
Write-Host "Unit test executables are in: $script:OutputDir" -ForegroundColor Cyan
exit 0
