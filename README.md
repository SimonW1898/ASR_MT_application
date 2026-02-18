# Streaming ASR→MT gRPC Demo (Application)

Real-time speech-to-text plus translation demo with a GUI client and gRPC server.

## Fast overview

- Windows-focused demo workflow using PowerShell launch scripts.
- PySide6 client streams WAV or microphone audio to a gRPC service.
- Server handles segmentation, preprocessing, ASR, MT, and JSONL logging.
- Reusable pipeline components live in `streaming/`; demo app code lives in `grpc_demo/`.

## Quick start

```powershell
./run_grpc_demo.ps1
```

Stop all launcher-managed processes:

```powershell
./stop_grpc_demo.ps1
```

If port `50051` is already in use:

```powershell
./run_grpc_demo.ps1 -KillExistingPortProcess
```

Manual two-terminal startup (same environment):

```powershell
# Terminal 1
.\venv311\Scripts\python.exe -m grpc_demo.server.server_main

# Terminal 2
.\venv311\Scripts\python.exe -m grpc_demo.client.app --config grpc_demo/configs/demo.yaml
```

For full step-by-step data setup, see [docs/data.md](docs/data.md).


## Documentation

Use docs for all setup and operational detail (including FFmpeg installation):

- [Docs Index](docs/index.md)
- [Setup (fresh machine)](docs/setup_fresh_machine.md)
- [Runtime Guide](docs/runtime.md)
- [Configuration](docs/configuration.md)
- [gRPC Demo Notes](docs/grpc_demo.md)
- [Model Cache Strategy](docs/model_cache.md)
- [Data Guide](docs/data.md)
- [Model/Data Licensing](docs/model_data_licensing.md)
- [PowerShell Scripts Reference](docs/scripts.md)
- [Troubleshooting](docs/troubleshooting.md)

## Repository structure

```text
application/
├─ README.md                        # project overview and entry point
├─ .gitignore                       # local/generated files excluded from git
├─ requirements-py311.txt           # runtime Python dependencies
├─ requirements-eval.txt            # evaluation/data-tooling dependencies
├─ run_grpc_demo.ps1                # starts server + GUI demo flow
├─ stop_grpc_demo.ps1               # stops launcher-managed demo processes
├─ check_env.ps1                    # environment/preflight checks
├─ prepare_model_cache.ps1          # warm up/download model cache
├─ context.txt                      # local Copilot/context helper file
├─ docs/                            # centralized project documentation
│  ├─ index.md                      # docs entry index
│  ├─ setup_fresh_machine.md        # full setup guide (incl. FFmpeg)
│  ├─ runtime.md                    # runtime and execution behavior
│  ├─ configuration.md              # config options and defaults
│  ├─ grpc_demo.md                  # demo-specific notes
│  ├─ model_cache.md                # model cache strategy and usage
│  ├─ data.md                       # data pipeline and dataset handling
│  ├─ model_data_licensing.md       # model/data source and license summary
│  ├─ scripts.md                    # script reference
│  ├─ modules.md                    # module-level overview
│  ├─ architecture.md               # architecture notes
│  ├─ portability.md                # portability roadmap
│  ├─ troubleshooting.md            # known issues and fixes
│  └─ changelog.md                  # release notes and changes
├─ grpc_demo/                       # demo application package
│  ├─ client/                       # PySide6 client, audio sources, UI flow
│  ├─ server/                       # gRPC service + processing pipeline
│  ├─ configs/                      # demo/runtime/model plan configs
│  ├─ proto/                        # protobuf source definitions
│  └─ proto_gen/                    # generated protobuf code
├─ streaming/                       # reusable streaming/ASR/MT components
│  ├─ asr/                          # ASR adapters and logic
│  ├─ mt/                           # machine translation adapters
│  ├─ buffers/                      # buffering and segment windowing
│  ├─ core/                         # shared interfaces/base abstractions
│  ├─ sinks/                        # output sinks/log sinks
│  ├─ sources/                      # input sources
│  ├─ models.py                     # shared model-related types
│  ├─ preprocess.py                 # preprocessing logic
│  └─ realtime.py                   # realtime orchestration
├─ scripts/                         # non-runtime helper tooling
│  ├─ data_manage/                  # dataset build/stream metadata scripts
│  └─ eval/                         # evaluation helper scripts
├─ data/                            # local data artifacts
│  ├─ raw/                          # raw/processed dataset material
│  └─ stream/                       # demo wav/json stream files
├─ model_cache/                     # local Hugging Face model snapshots
│  └─ hf_snapshots/                 # cached model repositories
├─ logs/                            # JSONL logs from demo runs
├─ md_old/                          # archived legacy docs
├─ venv311/                         # local Python 3.11 environment
├─ venv-eval/                       # local eval/tooling environment
└─ cv-corpus-24.0-*.tar.gz          # local Common Voice tar archives (optional)
```
### Folder-by-folder guide

- `docs/`
	- Main documentation hub and stable wiki pages
	- Start at [docs/index.md](docs/index.md)
	- Folder-level README files are intentionally removed; docs live only under `docs/`

- `scripts/` (tooling only, non-shipped)
	- Internal data/eval utilities separated from runtime
	- `data_manage/`: CVSS/CommonVoice raw build + stream_with_meta builders
	- `eval/`: evaluation helpers
	- Full guide: [docs/scripts.md](docs/scripts.md)

- `grpc_demo/`
	- Demo-specific application layer
	- `client/`: PySide6 GUI, WAV/mic sources, gRPC client flow
	- `server/`: gRPC service, segmentation/processing, model manager, logging
	- `configs/`: runtime YAML + model cache plan
	- `proto/` and `proto_gen/`: protobuf source and generated stubs

- `streaming/`
	- Reusable pipeline components shared by demo/server
	- ASR, MT, buffering/merge, preprocess, core abstractions

- `data/stream/`
	- WAV demo assets and optional sidecar JSON references
	- Sidecar JSON (same base filename) powers reference translation window in GUI

- `logs/`
	- JSONL run outputs generated by server sessions

- `model_cache/`
	- Local Hugging Face snapshots for offline/portable demo runs

- `md_old/`
	- Legacy source docs kept as historical reference input

- `venv311/`
	- Local Python 3.11 virtual environment (machine-local)

### Root scripts and what they do

- `run_grpc_demo.ps1`
	- Starts server and GUI in separate terminals
	- Supports `-KillExistingPortProcess` for stale binds

- `stop_grpc_demo.ps1`
	- Stops tracked terminals and fallback server/client processes
	- Clears listener leftovers on port `50051` when found

- `check_env.ps1`
	- Verifies Python path, imports, FFmpeg, config file, and port state

- `prepare_model_cache.ps1`
	- Reads model plan and warms ASR/MT models for offline demo stability
