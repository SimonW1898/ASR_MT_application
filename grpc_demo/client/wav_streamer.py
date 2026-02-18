"""WAV streaming client adapter."""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from typing import Iterator, Tuple


@dataclass(frozen=True)
class WavFrame:
    pcm16: bytes
    t0: int
    t1: int
    frame_idx: int


class WavFileSource:
    """Stream PCM16 frames from a WAV file."""

    def __init__(
        self,
        wav_path: str,
        sample_rate: int,
        chunk_ms: int,
        realtime_simulation: bool = True,
    ):
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.realtime_simulation = realtime_simulation
        self.frame_samples = int(sample_rate * chunk_ms / 1000)

    def frames(self) -> Iterator[WavFrame]:
        with wave.open(self.wav_path, "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError("WAV must be mono")
            if wf.getsampwidth() != 2:
                raise ValueError("WAV must be 16-bit PCM")
            if wf.getframerate() != self.sample_rate:
                raise ValueError("WAV sample rate mismatch")

            frame_idx = 0
            sample_cursor = 0

            while True:
                raw = wf.readframes(self.frame_samples)
                if not raw:
                    break

                n_samples = len(raw) // 2
                t0 = sample_cursor
                t1 = sample_cursor + n_samples
                sample_cursor = t1

                yield WavFrame(pcm16=raw, t0=t0, t1=t1, frame_idx=frame_idx)
                frame_idx += 1

                if self.realtime_simulation:
                    time.sleep(self.chunk_ms / 1000.0)
