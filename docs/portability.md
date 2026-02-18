[Back to Docs Index](index.md)

# Portability Roadmap

Date base: 2026-02-12 plan, organized as execution phases.

## Current coverage

- gRPC boundary and bidirectional stream are implemented
- Server owns segmentation, merge, ASR/MT inference, and JSONL logs
- Client owns source capture, transport framing, and UI rendering
- Warm-up entrypoint exists (`grpc_demo.server.init_models`)
- One-command launcher and stopper exist

## Phases

### Phase A — Reliable startup

Completed hardening:

- `check_env.ps1` preflight checks
- startup bind precheck for `50051`
- optional stale-process cleanup via `-KillExistingPortProcess`

Acceptance:

- `./check_env.ps1` returns actionable pass/fail
- launcher does not fail silently on stale bind conflicts

### Phase B — Offline demo readiness

Focus:

- verify model load and short inference after cache prep
- make cache/offline readiness explicit in logs/docs

Acceptance:

- demo can run without re-downloading models after warm-up

### Phase C — Distribution readiness

Focus:

- freeze Python 3.11 dependency set
- optional bootstrap setup script for clone-to-run flow
- concise troubleshooting matrix

Acceptance:

- can move from clone to runnable demo with minimal manual steps

### Phase D — Optional server container path
This is only collection of ideas for future but not the focus of current work.
Focus:

- server-only Docker path
- native client remains PySide6 + FFmpeg
- mount cache and logs as volumes

Acceptance:

- containerized server reachable by host GUI client

[Back to Docs Index](index.md)
