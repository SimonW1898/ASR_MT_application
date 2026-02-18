"""Audio ingress adapter (frames -> segments)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from streaming.buffers.buffers import FixedWindowBuffer, SilenceTerminatedBuffer, VADBuffer
from streaming.core.types import Segment
from streaming.preprocess import clean_audio, PreprocessConfig
from streaming.sources.gate import RMSGate

from .session import SessionConfig


@dataclass
class IngressStats:
    frames_received: int = 0
    segments_emitted: int = 0
    bytes_received: int = 0
    samples_received: int = 0


class GrpcAudioIngress:
    """Audio ingress for gRPC PCM16 frames."""

    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        self.stats = IngressStats()
        self._buffer = self._create_buffer()
        self._preprocess_cfg = PreprocessConfig(mode="none")

    def push_pcm16(self, pcm16: bytes, t0: int, t1: int) -> Iterable[Segment]:
        self.stats.frames_received += 1
        self.stats.bytes_received += len(pcm16)

        samples = len(pcm16) // 2
        if t1 > t0:
            samples = t1 - t0
        self.stats.samples_received += samples

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        audio, _stats = clean_audio(
            audio,
            sr=self.cfg.audio.sample_rate,
            mode="none",
            config=self._preprocess_cfg,
        )

        segments = self._buffer.push(audio, t0, t1)
        self.stats.segments_emitted += len(segments)
        return segments

    def flush(self, t_end: int) -> List[Segment]:
        segments = self._buffer.flush(t_end)
        self.stats.segments_emitted += len(segments)
        return segments

    def _create_buffer(self):
        policy = self.cfg.policy
        gate = RMSGate(threshold=policy.threshold) if policy.threshold else None
        sr = self.cfg.audio.sample_rate

        if policy.policy_type == "fixed_window":
            return FixedWindowBuffer(sr=sr, win_s=policy.win_ms / 1000, gate=gate)
        if policy.policy_type == "silence_terminated":
            return SilenceTerminatedBuffer(
                sr=sr,
                min_s=policy.min_ms / 1000,
                max_s=policy.max_ms / 1000,
                silence_s=policy.silence_ms / 1000,
                gate=gate,
            )
        if policy.policy_type == "vad":
            return VADBuffer(
                sr=sr,
                pre_s=policy.pre_ms / 1000,
                post_s=policy.post_ms / 1000,
                min_speech_s=policy.min_ms / 1000,
                end_silence_s=policy.silence_ms / 1000,
                max_s=policy.max_ms / 1000,
                gate=gate,
            )

        raise ValueError(f"Unknown policy: {policy.policy_type}")
