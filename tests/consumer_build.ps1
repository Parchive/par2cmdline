#!/usr/bin/env pwsh
# Build a program against the public header and the static library the way an
# embedding application does: only the public include directory on the include
# path, and no config.h.

param(
    [string]$Platform = "x64",
    [string]$LibPath = "",
    [string]$CliLibPath = ""
)

$ErrorActionPreference = "Stop"

$ExecDir = (Get-Location).Path
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($env:srcdir -and $env:srcdir -ne ".") {
    $RootDir = Join-Path $ExecDir $env:srcdir
} else {
    $RootDir = Split-Path -Parent $ScriptDir
}
$IncludeDir = Join-Path $RootDir "include"
if (-not $LibPath) {
    $LibPath = Join-Path $ExecDir "libpar2.lib"
}
if (-not $CliLibPath) {
    $CliLibPath = Join-Path $ExecDir "par2cli.lib"
}

Write-Host "-------------------------------------------------------"
Write-Host "An application can build against the public header alone"
Write-Host "-------------------------------------------------------"

if (-not (Test-Path $LibPath) -or -not (Test-Path $CliLibPath)) {
    Write-Host "Skipping: the libraries have not been built."
    exit 77
}

function Find-VSEnvironment {
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

    if (-not (Test-Path $vsWhere)) {
        throw "Could not find vswhere.exe. Please ensure Visual Studio is installed."
    }

    $vsPath = & $vsWhere -latest -property installationPath
    if (-not $vsPath) {
        throw "Could not find Visual Studio installation."
    }

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

    $vcArch = switch ($Platform) {
        "x64" { "x64" }
        "Win32" { "x86" }
        "ARM64" { "arm64" }
        default { "x64" }
    }

    $batchContent = @"
@echo off
call "$VcVarsAll" $vcArch >nul 2>&1
$Command
"@

    $tempBatch = Join-Path $env:TEMP "par2_consumer_$(Get-Random).bat"
    Set-Content -Path $tempBatch -Value $batchContent -Encoding ASCII

    try {
        $output = & cmd.exe /c $tempBatch 2>&1
        return @{ Output = $output; ExitCode = $LASTEXITCODE }
    }
    finally {
        Remove-Item $tempBatch -Force -ErrorAction SilentlyContinue
    }
}

try {
    $vcvarsall = Find-VSEnvironment
}
catch {
    Write-Host "Skipping: $_"
    exit 0
}

$workdir = Join-Path $ExecDir "runconsumer_build"
if (Test-Path $workdir) {
    Remove-Item $workdir -Force -Recurse
}
New-Item -ItemType Directory -Path $workdir | Out-Null

Push-Location $workdir

try {
    # An application is free to include any one of the public headers on its own,
    # so each of them has to carry what it names.
    foreach ($header in Get-ChildItem -Path (Join-Path $IncludeDir "par2") -Filter "*.h") {
        $name = $header.Name
        Set-Content -Path "standalone.cpp" -Value "#include <par2/$name>" -Encoding ASCII

        # Deliberately no /I for the source tree and no /DHAVE_CONFIG_H.
        $result = Invoke-VCCommand -VcVarsAll $vcvarsall -Platform $Platform `
            -Command "cl.exe /nologo /EHsc /std:c++17 /I `"$IncludeDir`" /Zs standalone.cpp"

        if ($result.ExitCode -ne 0) {
            $result.Output | ForEach-Object { Write-Host "    $_" }
            Write-Host "ERROR: <par2/$name> does not compile on its own"
            exit 1
        }
    }

    $consumer = Join-Path $RootDir "tests\consumer.cpp"

    $result = Invoke-VCCommand -VcVarsAll $vcvarsall -Platform $Platform `
        -Command "cl.exe /nologo /EHsc /std:c++17 /I `"$IncludeDir`" `"$consumer`" /Fe:consumer.exe /link `"$CliLibPath`" `"$LibPath`""

    if ($result.ExitCode -ne 0) {
        $result.Output | ForEach-Object { Write-Host "    $_" }
        Write-Host "ERROR: could not build against <par2/libpar2.h>"
        exit 1
    }

    & ".\consumer.exe"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: the program built against <par2/libpar2.h> failed"
        exit 1
    }
}
finally {
    Pop-Location
    if (Test-Path $workdir) {
        Remove-Item $workdir -Force -Recurse -ErrorAction SilentlyContinue
    }
}

exit 0
