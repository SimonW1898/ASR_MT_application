"""Minimal CLI gRPC client for handshake testing."""

from __future__ import annotations

import argparse
import time
from typing import Iterator

import grpc
import yaml

try:
    from grpc_demo.proto_gen import asrmt_pb2, asrmt_pb2_grpc
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing generated proto code. Generate grpc_demo/proto_gen/*.py first."
    ) from exc

from .wav_streamer import WavFileSource

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50051


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_config(cfg: dict) -> asrmt_pb2.SessionConfig:
    audio = cfg.get("audio", {})
    policy = cfg.get("policy", {})
    merge = cfg.get("merge", {})
    mt = cfg.get("mt", {})
    models = cfg.get("models", {})
    runtime = cfg.get("runtime", {})
    input_mode = cfg.get("input_mode", {})

    policy_type = policy.get("type", "vad")
    silence_cfg = policy.get("silence_terminated", {})
    vad_cfg = policy.get("vad", {})
    fixed_cfg = policy.get("fixed_window", {})

    return asrmt_pb2.SessionConfig(
        audio=asrmt_pb2.AudioConfig(
            sample_rate=int(audio.get("sample_rate", 16000)),
            transport_chunk_ms=int(audio.get("transport_chunk_ms", 40)),
        ),
        policy=asrmt_pb2.PolicyConfig(
            type=policy_type,
            silence_terminated=asrmt_pb2.SilenceTerminatedConfig(
                min_ms=int(silence_cfg.get("min_ms", 400)),
                max_ms=int(silence_cfg.get("max_ms", 8000)),
                silence_ms=int(silence_cfg.get("silence_ms", 500)),
                threshold=float(silence_cfg.get("threshold", 0.01)),
            ),
            vad=asrmt_pb2.VADConfig(
                pre_ms=int(vad_cfg.get("pre_ms", 200)),
                post_ms=int(vad_cfg.get("post_ms", 200)),
                min_ms=int(vad_cfg.get("min_ms", 400)),
                silence_ms=int(vad_cfg.get("silence_ms", 500)),
                max_ms=int(vad_cfg.get("max_ms", 8000)),
                threshold=float(vad_cfg.get("threshold", 0.01)),
            ),
            fixed_window=asrmt_pb2.FixedWindowConfig(
                win_ms=int(fixed_cfg.get("win_ms", 2000)),
            ),
        ),
        merge=asrmt_pb2.MergeConfig(
            enabled=bool(merge.get("enabled", True)),
            tail_tokens=int(merge.get("cfg", {}).get("tail_tokens", 50)),
            head_tokens=int(merge.get("cfg", {}).get("head_tokens", 50)),
            min_match_tokens=int(merge.get("cfg", {}).get("min_match_tokens", 2)),
            max_search_shift=int(merge.get("cfg", {}).get("max_search_shift", 8)),
        ),
        mt=asrmt_pb2.MTConfig(
            mode=str(mt.get("mode", "segment")),
        ),
        models=asrmt_pb2.ModelConfig(
            asr_model_id=str(models.get("asr_model_id", "openai/whisper-small")),
            mt_model_id=str(models.get("mt_model_id", "Helsinki-NLP/opus-mt-en-de")),
            asr_language=str(models.get("asr_language", "en")),
        ),
        runtime=asrmt_pb2.RuntimeConfig(
            device=str(runtime.get("device", "auto")),
            offline=bool(runtime.get("offline", False)),
            cache_dir=str(runtime.get("cache_dir", "model_cache/hf_snapshots")),
            log_dir=str(runtime.get("log_dir", "logs")),
            grpc_host=str(runtime.get("grpc_host", "127.0.0.1")),
            grpc_port=int(runtime.get("grpc_port", 50051)),
        ),
        input_mode=asrmt_pb2.InputMode(
            use_microphone=bool(input_mode.get("use_microphone", True)),
            wav_path=str(input_mode.get("wav_path", "data/stream/demo.wav")),
            realtime_simulation=bool(input_mode.get("realtime_simulation", True)),
            device_hint=str(input_mode.get("device_hint", "")),
        ),
    )


def _request_iter(
    cfg_msg: asrmt_pb2.SessionConfig,
    wav_path: str,
    realtime_simulation: bool,
    handshake_only: bool,
) -> Iterator[asrmt_pb2.ClientMessage]:
    yield asrmt_pb2.ClientMessage(config=cfg_msg)
    time.sleep(0.1)

    if not handshake_only:
        audio_cfg = cfg_msg.audio
        source = WavFileSource(
            wav_path=wav_path,
            sample_rate=audio_cfg.sample_rate,
            chunk_ms=audio_cfg.transport_chunk_ms,
            realtime_simulation=realtime_simulation,
        )

        for frame in source.frames():
            yield asrmt_pb2.ClientMessage(
                audio=asrmt_pb2.AudioChunk(
                    pcm16=frame.pcm16,
                    t0=frame.t0,
                    t1=frame.t1,
                    frame_idx=frame.frame_idx,
                )
            )

    yield asrmt_pb2.ClientMessage(control=asrmt_pb2.Control(action="STOP"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR->MT gRPC test client")
    parser.add_argument("--config", default="grpc_demo/configs/demo.yaml", help="Config YAML")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host (overrides config)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port (overrides config)")
    parser.add_argument("--handshake-only", action="store_true", help="Only send config and STOP")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        cfg = _load_config(args.config)
    except (FileNotFoundError, yaml.YAMLError):
        cfg = {}
    cfg_msg = _build_config(cfg)

    input_mode = cfg.get("input_mode", {})
    runtime = cfg.get("runtime", {})
    wav_path = str(input_mode.get("wav_path", "data/stream/demo.wav"))
    realtime_simulation = bool(input_mode.get("realtime_simulation", True))
    host = str(runtime.get("grpc_host", DEFAULT_HOST)) if args.host == DEFAULT_HOST else args.host
    try:
        runtime_port = int(runtime.get("grpc_port", DEFAULT_PORT))
    except (TypeError, ValueError):
        runtime_port = DEFAULT_PORT
    port = runtime_port if args.port == DEFAULT_PORT else args.port

    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = asrmt_pb2_grpc.ASRMTServiceStub(channel)

    responses = stub.StreamSession(
        _request_iter(
            cfg_msg,
            wav_path=wav_path,
            realtime_simulation=realtime_simulation,
            handshake_only=args.handshake_only,
        )
    )

    for resp in responses:
        if resp.HasField("status"):
            status = resp.status
            print(f"[STATUS] {status.state}: {status.message}")
        if resp.HasField("segment"):
            print(f"[SEGMENT] {resp.segment.segment_id}")


if __name__ == "__main__":
    main()
