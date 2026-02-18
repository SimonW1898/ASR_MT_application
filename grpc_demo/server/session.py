"""Session config/state models for gRPC server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

try:
    from grpc_demo.proto_gen import asrmt_pb2
except Exception:  # pragma: no cover
    asrmt_pb2 = None


@dataclass(frozen=True)
class MergeCfg:
    enabled: bool
    tail_tokens: int
    head_tokens: int
    min_match_tokens: int
    max_search_shift: int


@dataclass(frozen=True)
class PolicyCfg:
    policy_type: str
    min_ms: int
    max_ms: int
    silence_ms: int
    threshold: float
    pre_ms: int
    post_ms: int
    win_ms: int


@dataclass(frozen=True)
class AudioCfg:
    sample_rate: int
    transport_chunk_ms: int


@dataclass(frozen=True)
class MTCfg:
    mode: str


@dataclass(frozen=True)
class ModelCfg:
    asr_model_id: str
    mt_model_id: str
    asr_language: str


@dataclass(frozen=True)
class RuntimeCfg:
    device: str
    offline: bool
    cache_dir: str
    log_dir: str
    grpc_host: str
    grpc_port: int


@dataclass(frozen=True)
class InputModeCfg:
    use_microphone: bool
    wav_path: str
    realtime_simulation: bool
    device_hint: str


@dataclass(frozen=True)
class SessionConfig:
    audio: AudioCfg
    policy: PolicyCfg
    merge: MergeCfg
    mt: MTCfg
    models: ModelCfg
    runtime: RuntimeCfg
    input_mode: InputModeCfg

    @staticmethod
    def from_proto(cfg: Any) -> "SessionConfig":
        audio = cfg.audio
        policy = cfg.policy
        merge = cfg.merge
        mt = cfg.mt
        models = cfg.models
        runtime = cfg.runtime
        input_mode = cfg.input_mode

        if policy.type == "silence_terminated":
            pol = policy.silence_terminated
            policy_cfg = PolicyCfg(
                policy_type=policy.type,
                min_ms=pol.min_ms,
                max_ms=pol.max_ms,
                silence_ms=pol.silence_ms,
                threshold=pol.threshold,
                pre_ms=0,
                post_ms=0,
                win_ms=0,
            )
        elif policy.type == "vad":
            pol = policy.vad
            policy_cfg = PolicyCfg(
                policy_type=policy.type,
                min_ms=pol.min_ms,
                max_ms=pol.max_ms,
                silence_ms=pol.silence_ms,
                threshold=pol.threshold,
                pre_ms=pol.pre_ms,
                post_ms=pol.post_ms,
                win_ms=0,
            )
        else:
            pol = policy.fixed_window
            policy_cfg = PolicyCfg(
                policy_type=policy.type,
                min_ms=0,
                max_ms=0,
                silence_ms=0,
                threshold=0.0,
                pre_ms=0,
                post_ms=0,
                win_ms=pol.win_ms,
            )

        merge_cfg = MergeCfg(
            enabled=merge.enabled,
            tail_tokens=merge.tail_tokens,
            head_tokens=merge.head_tokens,
            min_match_tokens=merge.min_match_tokens,
            max_search_shift=merge.max_search_shift,
        )

        return SessionConfig(
            audio=AudioCfg(
                sample_rate=audio.sample_rate,
                transport_chunk_ms=audio.transport_chunk_ms,
            ),
            policy=policy_cfg,
            merge=merge_cfg,
            mt=MTCfg(mode=mt.mode),
            models=ModelCfg(
                asr_model_id=models.asr_model_id,
                mt_model_id=models.mt_model_id,
                asr_language=models.asr_language,
            ),
            runtime=RuntimeCfg(
                device=runtime.device,
                offline=runtime.offline,
                cache_dir=runtime.cache_dir,
                log_dir=runtime.log_dir,
                grpc_host=runtime.grpc_host,
                grpc_port=runtime.grpc_port,
            ),
            input_mode=InputModeCfg(
                use_microphone=input_mode.use_microphone,
                wav_path=input_mode.wav_path,
                realtime_simulation=input_mode.realtime_simulation,
                device_hint=input_mode.device_hint,
            ),
        )


class SessionState:
    """Minimal session state for Phase 1."""

    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        self.ready = False
        self.last_t1: Optional[int] = None

    def mark_ready(self) -> None:
        self.ready = True

    def update_last_t1(self, t1: int) -> None:
        if t1 is not None:
            self.last_t1 = t1
