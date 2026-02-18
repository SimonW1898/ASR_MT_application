$Port = 50051

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot

$StatePath = Join-Path $ScriptRoot ".grpc_demo_processes.json"

$pids = @()

if (-not (Test-Path $StatePath)) {
    Write-Host "No state file found: $StatePath"
} else {
    try {
        $state = Get-Content -Raw -Path $StatePath | ConvertFrom-Json
    } catch {
        Write-Host "Failed to parse state file: $StatePath"
        Write-Host "Delete it manually and retry."
        exit 1
    }

    if ($state.server_terminal_pid) { $pids += [int]$state.server_terminal_pid }
    if ($state.client_terminal_pid) { $pids += [int]$state.client_terminal_pid }
}

$pids = $pids | Select-Object -Unique

$listener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($null -ne $listener) {
    $pids += [int]$listener.OwningProcess
}

$pythonServerPids = @()
$pythonClientPids = @()
try {
    $pythonServerPids = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'grpc_demo\.server\.server_main' } |
        Select-Object -ExpandProperty ProcessId

    $pythonClientPids = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'grpc_demo\.client\.app' } |
        Select-Object -ExpandProperty ProcessId

    $pythonClientPids += Get-CimInstance Win32_Process -Filter "name = 'pythonw.exe'" |
        Where-Object { $_.CommandLine -match 'grpc_demo\.client\.app' } |
        Select-Object -ExpandProperty ProcessId
} catch {
    $pythonServerPids = @()
    $pythonClientPids = @()
}

if ($pythonServerPids.Count -gt 0) {
    $pids += $pythonServerPids
}
if ($pythonClientPids.Count -gt 0) {
    $pids += $pythonClientPids
}

$pids = $pids | Sort-Object -Unique

if (-not $pids -or $pids.Count -eq 0) {
    Write-Host "Nothing to stop (no tracked terminals and no server listener on port $Port)."
    if (Test-Path $StatePath) {
        Remove-Item -Force $StatePath
        Write-Host "Removed state file: $StatePath"
    }
    exit 0
}

$stopped = @()
$missing = @()

foreach ($procId in $pids) {
    try {
        Stop-Process -Id $procId -Force -ErrorAction Stop
        $stopped += $procId
    } catch {
        $missing += $procId
    }
}

if ($stopped.Count -gt 0) {
    Write-Host "Stopped PIDs: $($stopped -join ', ')"
}
if ($missing.Count -gt 0) {
    Write-Host "Already not running or inaccessible: $($missing -join ', ')"
}

if (Test-Path $StatePath) {
    Remove-Item -Force $StatePath
    Write-Host "Removed state file: $StatePath"
}
