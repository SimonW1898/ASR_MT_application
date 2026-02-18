[Back to Docs Index](index.md)

# Module Reference

## `grpc_demo/client`

- `app.py`: PySide6 GUI entrypoint and session worker integration
- `ffmpeg_mic.py`: microphone capture adapter (FFmpeg/DirectShow)
- `wav_streamer.py`: WAV frame iterator with realtime simulation
- `grpc_client.py`: non-GUI gRPC client path

## `grpc_demo/server`

- `server_main.py`: gRPC service bootstrap and session lifecycle
- `session.py`: typed config and session state models
- `audio_ingress.py`: frame decode, preprocess, policy segmentation
- `processor.py`: ASR → merge → MT processing and timings
- `model_manager.py`: model lifecycle, device/offline/cache behavior
- `init_models.py`: warm-up/cache population entrypoint
- `logger.py`: JSONL sink for segment/session events

## `streaming`

Reusable modular pipeline components:

- `buffers`: policy logic and transcript merge helpers
- `asr`: ASR model wrappers
- `mt`: MT model wrappers
- `core`: shared types/config/cache utilities
- `preprocess.py`, `realtime.py`, and sink/source abstractions

[Back to Docs Index](index.md)
