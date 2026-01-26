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
    foreach ($exe in $testExes) {
        $exePath = Join-Path $script:OutputDir $exe
        if (Test-Path $exePath) {
            Remove-Item $exePath -Force
            Write-Host "  Removed: $exe"
        }
    }
    # Clean object files
    Get-ChildItem -Path $script:OutputDir -Filter "*_test.obj" -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item $_.FullName -Force
        Write-Host "  Removed: $($_.Name)"
    }
    Write-Host ""
}

# Ensure output directory exists
if (-not (Test-Path $script:OutputDir)) {
    New-Item -ItemType Directory -Path $script:OutputDir -Force | Out-Null
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

# Unit test definitions
# Each test has: Name, Sources (relative to src/), Dependencies (relative to src/), NeedsLibpar2
$unitTests = @(
    @{
        Name = "letype_test"
        Sources = @("letype_test.cpp")
        Dependencies = @()
        NeedsLibpar2 = $false
    },
    @{
        Name = "crc_test"
        Sources = @("crc_test.cpp", "crc.cpp")
        Dependencies = @()
        NeedsLibpar2 = $false
    },
    @{
        Name = "md5_test"
        Sources = @("md5_test.cpp", "md5.cpp")
        Dependencies = @()
        NeedsLibpar2 = $false
    },
    @{
        Name = "galois_test"
        Sources = @("galois_test.cpp", "galois.cpp")
        Dependencies = @()
        NeedsLibpar2 = $false
    },
    @{
        Name = "reedsolomon_test"
        Sources = @("reedsolomon_test.cpp", "reedsolomon.cpp", "galois.cpp")
        Dependencies = @()
        NeedsLibpar2 = $false
    },
    @{
        Name = "utf8_test"
        Sources = @("utf8_test.cpp", "utf8.cpp")
        Dependencies = @()
        NeedsLibpar2 = $false
    },
    @{
        Name = "diskfile_test"
        Sources = @("diskfile_test.cpp")
        Dependencies = @()
        NeedsLibpar2 = $true
    },
    @{
        Name = "libpar2_test"
        Sources = @("libpar2_test.cpp")
        Dependencies = @()
        NeedsLibpar2 = $true
    },
    @{
        Name = "commandline_test"
        Sources = @("commandline_test.cpp")
        Dependencies = @()
        NeedsLibpar2 = $true
    },
    @{
        Name = "descriptionpacket_test"
        Sources = @("descriptionpacket_test.cpp")
        Dependencies = @()
        NeedsLibpar2 = $true
    },
    @{
        Name = "criticalpacket_test"
        Sources = @("criticalpacket_test.cpp")
        Dependencies = @()
        NeedsLibpar2 = $true
    }
)

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

# Get all library object files for linking tests that need libpar2
$libpar2Sources = @(
    "crc.cpp", "creatorpacket.cpp", "criticalpacket.cpp", "datablock.cpp",
    "descriptionpacket.cpp", "diskfile.cpp", "filechecksummer.cpp", "galois.cpp",
    "libpar2.cpp", "mainpacket.cpp", "md5.cpp", "par1fileformat.cpp",
    "par1repairer.cpp", "par1repairersourcefile.cpp", "par2creator.cpp",
    "par2creatorsourcefile.cpp", "par2fileformat.cpp", "par2repairer.cpp",
    "par2repairersourcefile.cpp", "recoverypacket.cpp", "reedsolomon.cpp",
    "verificationhashtable.cpp", "verificationpacket.cpp", "utf8.cpp",
    "commandline.cpp"
)

# Compiler flags
$debugFlags = "/Od /MDd /Zi /RTC1"
$releaseFlags = "/O2 /MD /DNDEBUG"
$commonFlags = "/EHsc /W3 /nologo /std:c++14 /DWIN32 /D_CONSOLE /DUNICODE /D_UNICODE /DPACKAGE=\`"par2cmdline\`" /DVERSION=\`"1.1.0\`""

$compilerFlags = if ($Configuration -eq "Debug") { "$commonFlags $debugFlags" } else { "$commonFlags $releaseFlags" }

$script:BuildSuccess = 0
$script:BuildFailed = 0

# Build each unit test
foreach ($test in $unitTests) {
    $testName = $test.Name
    $exePath = Join-Path $script:OutputDir "$testName.exe"
    
    Write-Host "Building: $testName" -ForegroundColor Cyan
    
    # Build source file list
    $sourceFiles = @()
    foreach ($src in $test.Sources) {
        $sourceFiles += """$(Join-Path $script:SrcDir $src)"""
    }
    
    # For tests that need libpar2, we need to include all library sources
    if ($test.NeedsLibpar2) {
        foreach ($libSrc in $libpar2Sources) {
            $srcPath = Join-Path $script:SrcDir $libSrc
            # Don't include if already in test sources
            $alreadyIncluded = $test.Sources | Where-Object { $_ -eq $libSrc }
            if (-not $alreadyIncluded -and (Test-Path $srcPath)) {
                $sourceFiles += """$srcPath"""
            }
        }
    }
    
    $sourceList = $sourceFiles -join " "
    
    # Build command
    $clCommand = "cl.exe $compilerFlags /I""$script:SrcDir"" /Fe""$exePath"" /Fo""$script:OutputDir\\"" $sourceList"
    
    if ($Verbose) {
        Write-Host "  Command: $clCommand" -ForegroundColor Gray
    }
    
    $result = Invoke-VCCommand -VcVarsAll $vcvarsall -Platform $Platform -Command $clCommand
    
    if ($result.ExitCode -eq 0) {
        Write-Host "  SUCCESS: $testName.exe" -ForegroundColor Green
        $script:BuildSuccess++
    }
    else {
        Write-Host "  FAILED: $testName.exe" -ForegroundColor Red
        if ($Verbose -or $true) {
            $result.Output | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
        }
        $script:BuildFailed++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Succeeded: $script:BuildSuccess" -ForegroundColor Green
Write-Host "Failed:    $script:BuildFailed" -ForegroundColor $(if ($script:BuildFailed -gt 0) { "Red" } else { "Green" })
Write-Host ""

if ($script:BuildFailed -gt 0) {
    Write-Host "Some unit tests failed to build. Check the output above for details." -ForegroundColor Yellow
    exit 1
}

Write-Host "All unit tests built successfully!" -ForegroundColor Green
Write-Host "Unit test executables are in: $script:OutputDir" -ForegroundColor Cyan
exit 0
