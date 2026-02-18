[Back to Docs Index](index.md)

# Changelog / Release Notes
## Current Goals:
- Include good preprocessing for noise reduction and better speach extraction



## 2026-02-16

### Changed

- GUI behavior in `grpc_demo.client.app`:
	- Session outputs are cleared immediately on **Start** (not delayed until first segment).
	- WAV-based auto-selection added for source/target language and model dropdown defaults.
- Model plan updates:
	- Added operational pairs for `ar-en`, `ja-en`, `fa-en`.
	- Switched Farsi MT recommendation to `facebook/m2m100_418M`.
- Cache warmup flow hardening:
	- `prepare_model_cache.ps1` now triggers warmups via explicit online mode support in `grpc_demo.server.init_models`.
- Segmentation defaults:
	- Demo policy default switched to VAD in config/runtime fallbacks.
- VAD buffering fix in `streaming/buffers/buffers.py`:
	- Real pre-roll from preceding audio (no synthetic zero-padding).
	- Real post-roll handling.
	- True speech-duration tracking for `min_speech_s` and false-start discard behavior.

### Documentation

- Removed folder-level README files in favor of centralized docs under `docs/`.
- Kept repository-level documentation entry at `README.md`.

## 2026-02-12

### Added

- Standalone reusable `streaming` package under `application/streaming`
- gRPC demo scaffold under `grpc_demo` (proto, server/client structure, configs)
- WAV streaming transport and segmentation pipeline with JSONL logging
- PySide6 GUI client path (`grpc_demo.client.app`) with live panes and latency metrics
- FFmpeg microphone adapter (`grpc_demo.client.ffmpeg_mic`)
- Environment/preflight script `check_env.ps1`
- Model cache prep flow `prepare_model_cache.ps1` + `model_cache_plan.json`
- Fresh-clone operational runbook and launcher/stopper workflow hardening

### Changed

- Processor moved from placeholders to model-backed ASR/MT path
- Server startup and client flow remain config-first with YAML runtime defaults
- Merge ownership and segmentation policy ownership remain server-side by design

### Verified

- End-to-end server/client runs with segment emission and JSONL outputs in `../logs`
- Warm-up entrypoint `grpc_demo.server.init_models` executes and begins cache population

## Next

- Strengthen explicit offline-ready inference verification after cache prep
- Freeze dependency set for distribution readiness
- Keep docs synchronized with script behavior as launch flow evolves

[Next: Model/Data Licensing](model_data_licensing.md)

[Back to Docs Index](index.md)
