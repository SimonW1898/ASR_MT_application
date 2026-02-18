[Back to Docs Index](index.md)

# gRPC Demo Notes

Focused notes for the `grpc_demo/` directory.

## Source of truth

The main operational guide is in the repository root:

- [../README.md](../README.md)

## Local map

- `proto/asrmt.proto` — protobuf source
- `proto_gen/` — generated stubs
- `configs/demo.yaml` — runtime config
- `client/` — GUI and client-side streaming
- `server/` — gRPC service and processing pipeline

## Documentation policy

Keep detailed setup/run/config docs in `docs/` to avoid drift.

- Folder-level `README` files under `grpc_demo/` are intentionally removed.
- Use the following pages for operations:
	- [Runtime Guide](runtime.md)
	- [Configuration](configuration.md)
	- [Scripts Guide](scripts.md)
	- [Troubleshooting](troubleshooting.md)

## Recent behavior updates

- GUI clears transcript/event output immediately when a new session starts.
- GUI can infer language/model defaults from WAV naming conventions (for example `fa-en`, `ja-en`, `ar-en`).
- Model selection is driven by `configs/model_cache_plan.json`.

## GUI controls quick map

- `Source`: ASR language selection
- `Target`: MT target language
- `ASR`: source-language-filtered ASR model list
- `MT`: source-target-filtered MT model list
- `Input file`: WAV picker for offline/replay sessions
- `Microphone`: live capture toggle

## Session test checklist

Before tests:

- Run `./check_env.ps1`
- Run `./prepare_model_cache.ps1` when models changed

Per test session:

1. Select source/target/model combo
2. Start session and wait for READY
3. Verify segment updates in ASR/MT panes
4. Stop session and verify STOPPED status
5. Inspect JSONL output in `../logs/`

Minimum pair coverage:

- `ar -> en`
- `fa -> en`
- `ja -> en`

[Back to Docs Index](index.md)
