[Back to Docs Index](index.md)

# Runtime Guide

Runtime usage for the gRPC demo UI + server.

## Scope

- Run the gRPC demo (server + GUI)
- Validate machine/runtime prerequisites
- Warm model cache for stable offline demos

## Run

From `application/`:

```powershell
./check_env.ps1
./prepare_model_cache.ps1 -DryRun
./prepare_model_cache.ps1
./run_grpc_demo.ps1
```

For alternate config:

```powershell
./run_grpc_demo.ps1 -Config grpc_demo/configs/demo.yaml
```

Stop demo:

```powershell
./stop_grpc_demo.ps1
```

## Manual startup (two terminals)

```powershell
# Terminal 1
.\venv311\Scripts\python.exe -m grpc_demo.server.server_main

# Terminal 2
.\venv311\Scripts\python.exe -m grpc_demo.client.app --config grpc_demo/configs/demo.yaml
```

## Runtime folders

- `grpc_demo/` — client/server/proto/config runtime code
- `streaming/` — shared ASR/MT/buffering modules
- `logs/` — session logs
- `data/stream/` — demo WAV assets used by GUI

## GUI workflow (recommended)

1. Select config path (or keep default).
2. Choose input mode:
	 - Microphone (`use_microphone=true`)
	 - WAV (`use_microphone=false`, select WAV file)
3. Choose source language.
4. Choose target language.
5. Pick ASR and MT models from filtered dropdowns.
6. Press **Start** and confirm immediate output reset + READY status.
7. Stop with **Stop** and validate final session status.

## Suggested validation cases

- **WAV sanity (stable)**
	- `ar -> en` with Arabic demo WAV
	- `fa -> en` with Persian/Farsi demo WAV
	- `ja -> en` with Japanese demo WAV
	- Expect: non-empty ASR, non-empty MT, no fallback-to-dummy logs

- **Microphone sanity (live)**
	- Speak 2-3 short sentences with pauses
	- Expect: segmentation triggers on pauses, no endless repeated output

- **Offline cache sanity**
	- Keep `runtime.offline: true`
	- Restart demo and rerun one pair
	- Expect: session initializes without new online downloads

[Back to Docs Index](index.md)
