[Back to Docs Index](index.md)

# Setup (Fresh Machine)

Goal: clone repo, prepare local model cache, and run demo with minimal manual steps.

## 1) Clone and open

```powershell
git clone <your-repo-url>
cd application
```

## 2) One-time script permission (PowerShell)

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
Unblock-File .\run_grpc_demo.ps1
Unblock-File .\stop_grpc_demo.ps1
Unblock-File .\check_env.ps1
Unblock-File .\prepare_model_cache.ps1
```

## 3) Create/refresh `venv311`

```powershell
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-py311.txt
```

## 4) Preflight checks

```powershell
./check_env.ps1
```

Expected checks include:

- Python executable under `venv311`
- Imports (`grpc`, `protobuf`, `PySide6`, `torch`, `transformers`, `yaml`)
- FFmpeg on `PATH`
- Config file presence
- Bind status for port `50051`

## 5) Prepare local model cache (recommended)

Preview plan:

```powershell
./prepare_model_cache.ps1 -DryRun
```

Populate cache:

```powershell
./prepare_model_cache.ps1
```

Notes:

- This reduces dependence on live Hugging Face downloads.
- Cache plan source: `grpc_demo/configs/model_cache_plan.json`
- Cache location defaults to `runtime.cache_dir` in `grpc_demo/configs/demo.yaml`
- Keep `runtime.offline: true` for strict offline runs after cache is prepared
- Current multilingual demo plan includes:
	- ASR source languages: `en`, `ar`, `fa`, `ja`
	- MT pairs: `en-de`, `de-en`, `de-fr`, `ar-en`, `ja-en`, `fa-en`
	- Farsi MT defaults to `facebook/m2m100_418M`

## 6) Run demo (one command)

```powershell
./run_grpc_demo.ps1
```

This opens server + GUI in separate terminals and uses the same config passed by `-Config`.

If `50051` is occupied:

```powershell
./run_grpc_demo.ps1 -KillExistingPortProcess
```

## 7) Stop demo

```powershell
./stop_grpc_demo.ps1
```

## 8) First GUI validation (recommended)

1. Start demo (`./run_grpc_demo.ps1`).
2. In GUI, pick source/target/model combination and WAV or microphone mode.
3. Verify a session reaches `READY` quickly.
4. Run one short test per target pair you plan to use:
	- `ar -> en`
	- `fa -> en`
	- `ja -> en`
5. Confirm merged ASR/MT panes update and logs are written under `logs/`.

## 9) Manual two-terminal startup

```powershell
# Terminal 1
.\venv311\Scripts\python.exe -m grpc_demo.server.server_main

# Terminal 2
.\venv311\Scripts\python.exe -m grpc_demo.client.app --config grpc_demo/configs/demo.yaml
```

[Back to Docs Index](index.md)
