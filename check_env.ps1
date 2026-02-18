param(
    [string]$Config = "grpc_demo/configs/demo.yaml"
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot

$failures = @()

function Write-Check([string]$Label, [bool]$Ok, [string]$Detail) {
    if ($Ok) {
        Write-Host "[OK ] $Label - $Detail"
    } else {
        Write-Host "[ERR] $Label - $Detail"
    }
}

$Py = Join-Path $ScriptRoot "venv311\Scripts\python.exe"
$pyExists = Test-Path $Py
Write-Check "Python" $pyExists $Py
if (-not $pyExists) { $failures += "Missing python at $Py" }

$configPath = Join-Path $ScriptRoot $Config
$configExists = Test-Path $configPath
Write-Check "Config" $configExists $configPath
if (-not $configExists) { $failures += "Missing config at $configPath" }

if ($pyExists) {
    $imports = @(
        "import grpc, google.protobuf; print('grpc/protobuf ok')",
        "import PySide6; print('PySide6 ok')",
        "import torch; print('torch', torch.__version__)",
        "import transformers; print('transformers', transformers.__version__)",
        "import yaml; print('yaml ok')"
    )

    foreach ($snippet in $imports) {
        & $Py -c $snippet *> $null
        $ok = $LASTEXITCODE -eq 0
        Write-Check "Import" $ok $snippet
        if (-not $ok) { $failures += "Import failed: $snippet" }
    }
}

$ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffmpegOk = $null -ne $ffmpegCmd
Write-Check "FFmpeg on PATH" $ffmpegOk "ffmpeg"
if (-not $ffmpegOk) { $failures += "FFmpeg not found on PATH" }

if ($ffmpegOk) {
    ffmpeg -hide_banner -version *> $null
    $ok = $LASTEXITCODE -eq 0
    Write-Check "FFmpeg runnable" $ok "ffmpeg -version"
    if (-not $ok) { $failures += "ffmpeg executable exists but failed to run" }
}

$conn = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 50051 -State Listen -ErrorAction SilentlyContinue
if ($null -eq $conn) {
    Write-Check "Port 50051" $true "free"
} else {
    $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
    Write-Check "Port 50051" $false "in use by PID $($conn.OwningProcess) ($($proc.ProcessName))"
    $failures += "Port 50051 already in use"
}

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "Environment check passed."
    exit 0
}

Write-Host "Environment check failed:" 
$failures | ForEach-Object { Write-Host " - $_" }
exit 1
