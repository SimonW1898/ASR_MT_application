"""Warm-up model initialization script (placeholder)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .model_manager import ModelManager
from .session import (
    AudioCfg,
    InputModeCfg,
    MergeCfg,
    ModelCfg,
    MTCfg,
    PolicyCfg,
    RuntimeCfg,
    SessionConfig,
)


def _load_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def _build_session_config(cfg: dict) -> SessionConfig:
    audio = cfg.get("audio", {})
    policy = cfg.get("policy", {})
    merge = cfg.get("merge", {})
    mt = cfg.get("mt", {})
    models = cfg.get("models", {})
    runtime = cfg.get("runtime", {})
    input_mode = cfg.get("input_mode", {})

    policy_type = str(policy.get("type", "vad"))
    silence_cfg = policy.get("silence_terminated", {})
    vad_cfg = policy.get("vad", {})
    fixed_cfg = policy.get("fixed_window", {})

    if policy_type == "vad":
        policy_cfg = PolicyCfg(
            policy_type=policy_type,
            min_ms=int(vad_cfg.get("min_ms", 400)),
            max_ms=int(vad_cfg.get("max_ms", 8000)),
            silence_ms=int(vad_cfg.get("silence_ms", 500)),
            threshold=float(vad_cfg.get("threshold", 0.01)),
            pre_ms=int(vad_cfg.get("pre_ms", 200)),
            post_ms=int(vad_cfg.get("post_ms", 200)),
            win_ms=0,
        )
    elif policy_type == "fixed_window":
        policy_cfg = PolicyCfg(
            policy_type=policy_type,
            min_ms=0,
            max_ms=0,
            silence_ms=0,
            threshold=0.0,
            pre_ms=0,
            post_ms=0,
            win_ms=int(fixed_cfg.get("win_ms", 2000)),
        )
    else:
        policy_cfg = PolicyCfg(
            policy_type="silence_terminated",
            min_ms=int(silence_cfg.get("min_ms", 400)),
            max_ms=int(silence_cfg.get("max_ms", 8000)),
            silence_ms=int(silence_cfg.get("silence_ms", 500)),
            threshold=float(silence_cfg.get("threshold", 0.01)),
            pre_ms=0,
            post_ms=0,
            win_ms=0,
        )

    return SessionConfig(
        audio=AudioCfg(
            sample_rate=int(audio.get("sample_rate", 16000)),
            transport_chunk_ms=int(audio.get("transport_chunk_ms", 40)),
        ),
        policy=policy_cfg,
        merge=MergeCfg(
            enabled=bool(merge.get("enabled", True)),
            tail_tokens=int(merge.get("cfg", {}).get("tail_tokens", 50)),
            head_tokens=int(merge.get("cfg", {}).get("head_tokens", 50)),
            min_match_tokens=int(merge.get("cfg", {}).get("min_match_tokens", 2)),
            max_search_shift=int(merge.get("cfg", {}).get("max_search_shift", 8)),
        ),
        mt=MTCfg(mode=str(mt.get("mode", "segment"))),
        models=ModelCfg(
            asr_model_id=str(models.get("asr_model_id", "openai/whisper-small")),
            mt_model_id=str(models.get("mt_model_id", "Helsinki-NLP/opus-mt-en-de")),
            asr_language=str(models.get("asr_language", "en")),
        ),
        runtime=RuntimeCfg(
            device=str(runtime.get("device", "auto")),
            offline=bool(runtime.get("offline", False)),
            cache_dir=str(runtime.get("cache_dir", "model_cache/hf_snapshots")),
            log_dir=str(runtime.get("log_dir", "logs")),
            grpc_host=str(runtime.get("grpc_host", "127.0.0.1")),
            grpc_port=int(runtime.get("grpc_port", 50051)),
        ),
        input_mode=InputModeCfg(
            use_microphone=bool(input_mode.get("use_microphone", True)),
            wav_path=str(input_mode.get("wav_path", "data/stream/demo.wav")),
            realtime_simulation=bool(input_mode.get("realtime_simulation", True)),
            device_hint=str(input_mode.get("device_hint", "")),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up ASR/MT models from config")
    parser.add_argument(
        "--config",
        default="grpc_demo/configs/demo.yaml",
        help="Config YAML path",
    )
    parser.add_argument("--asr-only", action="store_true", help="Warm up only ASR")
    parser.add_argument("--mt-only", action="store_true", help="Warm up only MT")
    parser.add_argument("--asr-model", default=None, help="Override ASR model id")
    parser.add_argument("--mt-model", default=None, help="Override MT model id")
    parser.add_argument("--asr-language", default=None, help="Override ASR language")
    parser.add_argument(
        "--force-online",
        action="store_true",
        help="Force online warmup (runtime.offline=False) to download/cache missing models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _build_session_config(_load_yaml(args.config))

    if args.asr_model is not None:
        cfg = SessionConfig(
            audio=cfg.audio,
            policy=cfg.policy,
            merge=cfg.merge,
            mt=cfg.mt,
            models=ModelCfg(
                asr_model_id=args.asr_model,
                mt_model_id=cfg.models.mt_model_id,
                asr_language=cfg.models.asr_language,
            ),
            runtime=cfg.runtime,
            input_mode=cfg.input_mode,
        )
    if args.mt_model is not None:
        cfg = SessionConfig(
            audio=cfg.audio,
            policy=cfg.policy,
            merge=cfg.merge,
            mt=cfg.mt,
            models=ModelCfg(
                asr_model_id=cfg.models.asr_model_id,
                mt_model_id=args.mt_model,
                asr_language=cfg.models.asr_language,
            ),
            runtime=cfg.runtime,
            input_mode=cfg.input_mode,
        )
    if args.asr_language is not None:
        cfg = SessionConfig(
            audio=cfg.audio,
            policy=cfg.policy,
            merge=cfg.merge,
            mt=cfg.mt,
            models=ModelCfg(
                asr_model_id=cfg.models.asr_model_id,
                mt_model_id=cfg.models.mt_model_id,
                asr_language=args.asr_language,
            ),
            runtime=cfg.runtime,
            input_mode=cfg.input_mode,
        )
    if args.force_online:
        cfg = SessionConfig(
            audio=cfg.audio,
            policy=cfg.policy,
            merge=cfg.merge,
            mt=cfg.mt,
            models=cfg.models,
            runtime=RuntimeCfg(
                device=cfg.runtime.device,
                offline=False,
                cache_dir=cfg.runtime.cache_dir,
                log_dir=cfg.runtime.log_dir,
                grpc_host=cfg.runtime.grpc_host,
                grpc_port=cfg.runtime.grpc_port,
            ),
            input_mode=cfg.input_mode,
        )

    Path(cfg.runtime.cache_dir).mkdir(parents=True, exist_ok=True)

    manager = ModelManager(cfg)
    if args.asr_only and not args.mt_only:
        manager.load_asr()
        print("[init_models] ASR model warmed up")
    elif args.mt_only and not args.asr_only:
        manager.load_mt()
        print("[init_models] MT model warmed up")
    else:
        manager.warmup()
        print("[init_models] ASR + MT models warmed up")

    status = manager.status()
    print(
        "[init_models] status "
        f"asr_loaded={status.asr_loaded} mt_loaded={status.mt_loaded} "
        f"asr_model={status.asr_model_id} mt_model={status.mt_model_id}"
    )


if __name__ == "__main__":
    main()
