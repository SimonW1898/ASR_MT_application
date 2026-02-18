[Back to Docs Index](index.md)

# Configuration

Primary runtime configuration file:

- `../grpc_demo/configs/demo.yaml`

## Key sections

- `audio`: sample rate and transport chunk size
- `input_mode`: WAV/microphone mode, WAV path, realtime simulation, device hint
- `policy`: segmentation mode and policy parameters (`silence_terminated`, `vad`, `fixed_window`)
- `merge`: merge toggle and merge parameters
- `mt`: translation mode (`segment` or `merged`)
- `models`: ASR model, MT model, and ASR language
- `runtime`: device, offline mode, cache dir, log dir, gRPC host/port
- `ui`: persisted GUI convenience state (for example target language)

## Current defaults (demo)

- `policy.type`: `vad`
- `merge.enabled`: `false` (conservative default while tuning live microphone behavior)
- Typical active multilingual demo pairs include `en-de`, `de-en`, `ar-en`, `ja-en`, and `fa-en`
- Typical source languages: `de`, `en`, `ar`, `fa`, `ja`

## MT model note (Farsi)

- Farsi routing is configured to use `facebook/m2m100_418M` instead of OPUS for improved quality in current tests.
- Keep `model_cache_plan.json` and runtime YAML in sync when changing pair defaults.

## Operational notes

- GUI writes key selections back into this YAML on session start.
- GUI can infer source/target defaults from WAV naming patterns (for example `fa-en`, `ja-en`, `arabic_en`).
- Offline behavior is controlled by `runtime.offline`.
- Model choices are constrained by `../grpc_demo/configs/model_cache_plan.json`.

## Add new languages to the demo

Language availability in the GUI is driven by the cache plan plus YAML runtime selection.

1. Add ASR language/model entries in `../grpc_demo/configs/model_cache_plan.json` under `asr_models`.
2. Add MT pair entries in the same file under `mt_models` using `pair` format `source-target` (for example `en-ja`, `en-ar`).
3. Warm the new models:

```powershell
./prepare_model_cache.ps1 -DryRun
./prepare_model_cache.ps1
```

4. Keep `runtime.offline: true` in `../grpc_demo/configs/demo.yaml` for stable offline demos.
5. Start the GUI and select:
	- ASR source language
	- MT target language
	- ASR/MT models from the filtered dropdowns

The GUI language and model dropdowns are built from `model_cache_plan.json`, so new pairs become selectable after plan update and cache warm-up.

<!-- For WAV demos with visible references, place sidecar JSON next to the WAV (same base filename) and include `segments[].tgt_ref` (or `segments[].tgt_ref_<lang>` for language-specific references). -->

[Next: PowerShell Scripts Reference](scripts.md)
[Back to Docs Index](index.md)
