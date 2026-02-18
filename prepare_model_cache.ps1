param(
    [string]$Config = "grpc_demo/configs/demo.yaml",
    [string]$Plan = "grpc_demo/configs/model_cache_plan.json",
    [switch]$DryRun
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot

$Py = Join-Path $ScriptRoot "venv311\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "Python executable not found at: $Py"
}

Write-Host "Preparing local model cache..."
Write-Host "Config: $Config"
Write-Host "Plan  : $Plan"
Write-Host ""

$PlanPath = Join-Path $ScriptRoot $Plan
if (-not (Test-Path $PlanPath)) {
    throw "Model cache plan file not found: $PlanPath"
}

try {
    $planObj = Get-Content -Raw -Path $PlanPath | ConvertFrom-Json
} catch {
    throw "Failed to parse model cache plan JSON: $PlanPath"
}

if ($DryRun) {
    Write-Host "[DRY RUN] ASR models:"
    foreach ($entry in $planObj.asr_models) {
        Write-Host "  - $($entry.name): $($entry.model_id) (lang=$($entry.asr_language))"
    }
    Write-Host "[DRY RUN] MT models:"
    foreach ($entry in $planObj.mt_models) {
        Write-Host "  - $($entry.pair): $($entry.model_id)"
    }
    exit 0
}

foreach ($entry in $planObj.asr_models) {
    $asrModel = [string]$entry.model_id
    $asrLang = [string]$entry.asr_language
    Write-Host "[CACHE] Warming ASR -> $asrModel (lang=$asrLang)"
    & $Py -m grpc_demo.server.init_models --config $Config --asr-only --asr-model $asrModel --asr-language $asrLang --force-online
    if ($LASTEXITCODE -ne 0) {
        throw "ASR warmup failed for model $asrModel"
    }
}

foreach ($entry in $planObj.mt_models) {
    $pair = [string]$entry.pair
    $mtModel = [string]$entry.model_id
    Write-Host "[CACHE] Warming MT for $pair -> $mtModel"
    & $Py -m grpc_demo.server.init_models --config $Config --mt-only --mt-model $mtModel --force-online
    if ($LASTEXITCODE -ne 0) {
        throw "MT warmup failed for pair $pair"
    }
}

Write-Host ""
Write-Host "Model cache preparation completed."
