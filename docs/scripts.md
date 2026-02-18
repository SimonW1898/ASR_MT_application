[Back to Docs Index](index.md)

# PowerShell Scripts Reference

Runtime and tooling script reference for this repository.

## Script areas

- Root `*.ps1` scripts: runtime launch, stop, preflight, cache prep
- `scripts/data_manage/`: dataset and stream-building tooling
- `scripts/eval/`: evaluation utilities

## `run_grpc_demo.ps1`

- Starts server and GUI in separate terminals
- Writes state file `.grpc_demo_processes.json`
- Parameters:
  - `-Config` (default `grpc_demo/configs/demo.yaml`)
  - `-DryRun`
  - `-KillExistingPortProcess`

Typical usage:

```powershell
./run_grpc_demo.ps1
./run_grpc_demo.ps1 -KillExistingPortProcess
./run_grpc_demo.ps1 -DryRun
```

## `stop_grpc_demo.ps1`

- Reads `.grpc_demo_processes.json`
- Stops saved terminal PIDs
- Removes state file

```powershell
./stop_grpc_demo.ps1
```

## `check_env.ps1`

Preflight checks for:

- `venv311` Python path
- required imports (`grpc`, `protobuf`, `PySide6`, `torch`, `transformers`, `yaml`)
- FFmpeg on `PATH`
- config file presence
- port `50051` bind status

```powershell
./check_env.ps1
```

## `prepare_model_cache.ps1`

- Reads cache plan JSON
- Warms ASR models and MT models via `grpc_demo.server.init_models`
- Supports dry-run preview

```powershell
./prepare_model_cache.ps1 -DryRun
./prepare_model_cache.ps1
```

## Tooling data root WARNING: Just do this when CoVoST downloaded and need new evaluation Data

Tooling scripts under `application/scripts/` resolve paths from `__file__` and write to:

- `../data/` (parent of `application/`)

In this workspace layout that is:

- `c:/Users/Simon.Wenchel/Code/data/`

## Tooling quick start

From `application/`:

```powershell
python scripts/data_manage/download_cvss_raw.py --languages ar,fa,ja --min_matches 10 --max_per_split 20 --seed 123 --common_voice_tar .\cv-corpus-24.0-2025-12-05-ar.tar.gz
python scripts/data_manage/build_stream_with_meta.py --pairs ar-en,fa-en,ja-en --sentences-per-stream 5 --use-all-sentences --min-gap 0.2 --max-gap 0.6
```

Notes:

- `download_cvss_raw.py` is the active raw-data builder.
- `build_stream_with_meta.py` creates concatenated stream WAV/JSON artifacts under `data/stream/<pair>/`.

## Separate eval environment (recommended)

Use a dedicated eval venv for data/eval tooling to avoid mixing with runtime dependencies.
There are problems with `protobuf` package, that is needed for grpc and some evaluation tools, such as `unbabel-comet`.

```powershell
py -3.11 -m venv venv_eval
.\venv_eval\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-eval.txt
```

Notes:

- `requirements-eval.txt` intentionally excludes demo-runtime `grpcio` dependencies.
- Keep runtime flow on `venv311` and eval/data flow on `venv_eval` (or `venv-eval`).

[Next: Data Guidelines](data.md)
[Back to Docs Index](index.md)
