[Back to Docs Index](index.md)

# Architecture

This demo keeps a strict client/server split and reuses class-based `streaming` modules.

## System boundary

Client (GUI + audio capture/input) → gRPC stream → Server (policy + preprocess + ASR + merge + MT + logging) → gRPC stream → Client display.

## Responsibilities

### Client (`grpc_demo/client`)

- Owns PySide6 UI state and controls
- Owns source capture (microphone via FFmpeg, or WAV streaming)
- Owns transport framing and gRPC session lifecycle
- Modifies settings setup for a user session and sends it with first message as config
- Renders merged ASR, merged MT, and latency metrics

### Server (`grpc_demo/server`)

- Owns session state, segmentation policy buffers, and counters
- Owns preprocessing invocation and segment emission
- Owns model loading/inference (ASR and MT)
- Owns merge logic and JSONL logging

## Pipeline model

`AudioChunk` frames are transport units; server policy emits variable-length segments for processing.

Server processing sequence:

1. PCM16 → float32
2. `preprocess(audio)`
3. policy segmentation
4. ASR on segment
5. optional merge (`merge_transcripts_align`)
6. MT (`segment` mode by default)
7. emit gRPC result + JSONL record

## Reuse strategy

- Preserve existing `streaming` abstractions (buffers, ASR, MT, merge, core types)
- Add adapters at boundaries in `grpc_demo/server` and `grpc_demo/client`
- Keep pipeline composition stable instead of rewriting internals

## Session protocol (high level)

- First client message must be config
- Server sends status events (`LOADING`, `READY`, `STOPPED`, `ERROR`)
- Streaming audio follows `READY`
- `STOP` performs graceful flush/drain before close

[Next: Troubleshooting](troubleshooting.md)
[Back to Docs Index](index.md)
