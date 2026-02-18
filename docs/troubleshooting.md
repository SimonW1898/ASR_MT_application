[Back to Docs Index](index.md)

# Troubleshooting

## Port bind issues (`127.0.0.1:50051`)

- Stop launched terminals:

```powershell
./stop_grpc_demo.ps1
```

- Or start with automatic cleanup:

```powershell
./run_grpc_demo.ps1 -KillExistingPortProcess
```

## Import / dependency errors

- Activate `venv311`
- Reinstall dependencies:

```powershell
python -m pip install -r requirements-py311.txt
```

- Re-run preflight:

```powershell
./check_env.ps1
```

## FFmpeg microphone device issues

- Validate FFmpeg availability:

```powershell
ffmpeg -hide_banner -version
```

- List DirectShow devices:

```powershell
ffmpeg -list_devices true -f dshow -i dummy
```

- Set/update `input_mode.device_hint` in `../grpc_demo/configs/demo.yaml` or via GUI.

## Offline/cache behavior

- Preview cache plan:

```powershell
./prepare_model_cache.ps1 -DryRun
```

- Warm local cache:

```powershell
./prepare_model_cache.ps1
```

- Ensure `runtime.offline: true` in `../grpc_demo/configs/demo.yaml` for strict offline runs.

<!-- ## Whisper hallucinations / repeated filler text

Symptoms:

- Repeated phrases (for example subtitles-like boilerplate) in low-speech/noisy intervals

Mitigations (recommended order):

1. Verify language/model match (`models.asr_language` matches actual speech language).
2. Prefer WAV sanity runs before microphone runs.
3. Tighten VAD policy in `../grpc_demo/configs/demo.yaml`:
	- increase `policy.vad.min_ms`
	- increase `policy.vad.silence_ms`
	- increase `policy.vad.threshold`
4. Keep `merge.enabled: false` while diagnosing repetition.
5. Test with a larger Whisper model when available.

If quality is still poor in microphone mode, capture a short WAV sample and reproduce with file mode to separate microphone/noise issues from model issues. -->

[Next: Portability Roadmap](portability.md)
[Back to Docs Index](index.md)
