param(
    [string]$Config = "grpc_demo/configs/demo.yaml",
    [switch]$DryRun,
    [switch]$KillExistingPortProcess
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot
$StatePath = Join-Path $ScriptRoot ".grpc_demo_processes.json"

$Py = Join-Path $ScriptRoot "venv311\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Python executable not found at: $Py"
}

$ConfigPath = Join-Path $ScriptRoot $Config
if (-not (Test-Path $ConfigPath)) {
    throw "Config file not found: $ConfigPath"
}

$existingConn = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 50051 -State Listen -ErrorAction SilentlyContinue
if ($null -ne $existingConn) {
    $existingPid = $existingConn.OwningProcess
    $existingProc = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    Write-Host "[PRECHECK] Port 50051 already in use by PID $existingPid ($($existingProc.ProcessName))."

    if ($KillExistingPortProcess) {
        Write-Host "[PRECHECK] Stopping existing process on 50051..."
        Stop-Process -Id $existingPid -Force
        Start-Sleep -Milliseconds 400
    } else {
        throw "Port 50051 is already in use. Run ./stop_grpc_demo.ps1 or rerun with -KillExistingPortProcess."
    }
}

$ServerCmd = "Set-Location '$ScriptRoot'; & '$Py' -m grpc_demo.server.server_main --config '$Config'"
$ClientCmd = "Set-Location '$ScriptRoot'; & '$Py' -m grpc_demo.client.app --config '$Config'"

Write-Host "Project root : $ScriptRoot"
Write-Host "Python       : $Py"
Write-Host "Config       : $Config"
Write-Host ""
Write-Host "Launching server and GUI in separate terminals..."

if ($DryRun) {
    Write-Host "[DRY RUN] Server command: $ServerCmd"
    Write-Host "[DRY RUN] Client command: $ClientCmd"
    Write-Host "[DRY RUN] State file    : $StatePath"
    exit 0
}

$serverProc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-Command", $ServerCmd) -PassThru
Start-Sleep -Milliseconds 700
$clientProc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-Command", $ClientCmd) -PassThru

$state = [ordered]@{
    created_at = (Get-Date).ToString("o")
    script_root = $ScriptRoot
    config = $Config
    server_terminal_pid = $serverProc.Id
    client_terminal_pid = $clientProc.Id
}
$state | ConvertTo-Json | Set-Content -Path $StatePath -Encoding UTF8

Write-Host ""
Write-Host "Server terminal PID: $($serverProc.Id)"
Write-Host "Client terminal PID: $($clientProc.Id)"
Write-Host "State file         : $StatePath"
Write-Host ""
Write-Host "To stop later, run:"
Write-Host "  .\stop_grpc_demo.ps1"
