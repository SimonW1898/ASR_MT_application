[Back to Docs Index](index.md)

# Model Cache Strategy

Goal: reduce live-download dependence by pre-caching selected ASR and MT models.

## Cache plan

Plan file:

- `../grpc_demo/configs/model_cache_plan.json`

Current plan includes ASR models and selected MT language pairs.

Current notable entries:

- `ar-en`: `Helsinki-NLP/opus-mt-ar-en`
- `ja-en`: `Helsinki-NLP/opus-mt-ja-en`
- `fa-en`: `facebook/m2m100_418M`

## Add a new language pair

1. Open `../grpc_demo/configs/model_cache_plan.json`.
2. Add/confirm an ASR model entry for the source language in `asr_models`:

```json
{
	"name": "Whisper Small (English)",
	"model_id": "openai/whisper-small",
	"asr_language": "en"
}
```

3. Add MT entries for each direction you want in `mt_models`:

```json
{
	"pair": "en-ja",
	"model_id": "<valid-mt-model-id-for-en-ja>"
}
```

```json
{
	"pair": "en-ar",
	"model_id": "<valid-mt-model-id-for-en-ar>"
}
```

4. Run cache warm-up so models are available offline:

```powershell
./prepare_model_cache.ps1 -DryRun
./prepare_model_cache.ps1
```

5. Start the GUI and select the new source/target language pair.

If a model is not available for a pair, it will not appear in the GUI dropdowns.

After changing the plan, always run one GUI session per affected pair to verify no dummy fallback logs appear.

## Prepare cache

Preview cache actions:

```powershell
./prepare_model_cache.ps1 -DryRun
```

Run cache preparation:

```powershell
./prepare_model_cache.ps1
```

The script drives `grpc_demo.server.init_models` with per-model overrides from the plan.

## Offline toggle

After cache preparation, keep offline mode enabled in `../grpc_demo/configs/demo.yaml`:

- `runtime.offline: true`

This enforces local-cache execution for demo runs.

[Back to Docs Index](index.md)
