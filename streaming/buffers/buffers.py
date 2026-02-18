"""
Buffer policies for the streaming pipeline.

Controls when to emit audio segments for ASR processing.
Supports multiple strategies: wait-k sliding window, fixed window,
silence-terminated, and VAD-based buffering.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..core.types import Segment, DEFAULT_SR, DEFAULT_K_S, DEFAULT_HOP_S, DEFAULT_WIN_S
from ..core.ringbuffer import RingBuffer
from ..sources.gate import SpeechGate


class BufferPolicy(ABC):
    """Abstract interface for audio buffering strategies."""

    @property
    def has_overlap(self) -> bool:
        """Whether this buffer produces overlapping segments."""
        return False

    @abstractmethod
    def push(self, frame_audio: np.ndarray, t0: int, t1: int) -> List[Segment]:
        """Push audio frame and return any segments ready for ASR."""
        raise NotImplementedError()

    @abstractmethod
    def flush(self, t_end: int) -> List[Segment]:
        """Flush remaining audio at end of stream."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        """Reset buffer state."""
        raise NotImplementedError()


class WaitKSlidingWindowBuffer(BufferPolicy):
    """
    Wait-k sliding window buffer.

    Waits for k seconds of audio, then emits overlapping windows with
    configurable hop and window size.
    """

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        k_s: float = DEFAULT_K_S,
        hop_s: float = DEFAULT_HOP_S,
        win_s: float = DEFAULT_WIN_S,
    ):
        self.sr = sr
        self.k_samples = int(k_s * sr)
        self.hop_samples = int(hop_s * sr)
        self.win_samples = int(win_s * sr)

        self.ring = RingBuffer(self.win_samples)
        self.total_seen_samples = 0
        self.next_emit_at = self.k_samples
        self.started = False

        self.tags = {
            "k_s": k_s,
            "hop_s": hop_s,
            "win_s": win_s,
        }

    @property
    def has_overlap(self) -> bool:
        return self.hop_samples < self.win_samples

    def push(self, frame_audio: np.ndarray, t0: int, t1: int) -> List[Segment]:
        self.ring.append(frame_audio)
        self.total_seen_samples = t1

        segments = []
        while self.total_seen_samples >= self.next_emit_at:
            seg = self._emit_window(self.next_emit_at, reason="waitk_hop")
            segments.append(seg)
            self.next_emit_at += self.hop_samples
            self.started = True

        return segments

    def flush(self, t_end: int) -> List[Segment]:
        if self.ring.get_size() == 0:
            return []
        return [self._emit_window(t_end, reason="flush")]

    def reset(self) -> None:
        self.ring.clear()
        self.total_seen_samples = 0
        self.next_emit_at = self.k_samples
        self.started = False

    def _emit_window(self, t_end: int, reason: str) -> Segment:
        audio = self.ring.slice_last(min(self.win_samples, self.ring.get_size()))
        end_sample = t_end
        start_sample = max(0, end_sample - len(audio))

        return Segment(
            audio=audio,
            start_sample=start_sample,
            end_sample=end_sample,
            reason=reason,
            tags=self.tags.copy(),
        )


class FixedWindowBuffer(BufferPolicy):
    """
    Fixed window buffer.

    Accumulates audio until window size is reached, then emits.
    Optionally uses a speech gate to only buffer during speech.
    """

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        win_s: float = 5.0,
        gate: Optional[SpeechGate] = None,
    ):
        self.sr = sr
        self.win_samples = int(win_s * sr)
        self.gate = gate

        self.ring = RingBuffer(self.win_samples * 2)
        self.active = False
        self.seg_start_sample = 0
        self.seg_samples = 0

        self.tags = {"win_s": win_s}

    def push(self, frame_audio: np.ndarray, t0: int, t1: int) -> List[Segment]:
        segments = []

        if self.gate is not None:
            is_speech = self.gate.is_speech(frame_audio)
            if not is_speech and not self.active:
                return segments
            if is_speech and not self.active:
                self.active = True
                self.seg_start_sample = t0
                self.seg_samples = 0
        else:
            if not self.active:
                self.active = True
                self.seg_start_sample = t0
                self.seg_samples = 0

        self.ring.append(frame_audio)
        self.seg_samples += len(frame_audio)

        if self.seg_samples >= self.win_samples:
            seg = self._emit(t1, reason="fixed_window")
            segments.append(seg)
            self.ring.clear()
            self.seg_samples = 0
            self.active = False

        return segments

    def flush(self, t_end: int) -> List[Segment]:
        if self.seg_samples == 0:
            return []

        seg = self._emit(t_end, reason="flush")
        self.ring.clear()
        self.seg_samples = 0
        self.active = False
        return [seg]

    def reset(self) -> None:
        self.ring.clear()
        self.active = False
        self.seg_start_sample = 0
        self.seg_samples = 0

    def _emit(self, t_end: int, reason: str) -> Segment:
        audio = self.ring.get_all()
        return Segment(
            audio=audio,
            start_sample=self.seg_start_sample,
            end_sample=t_end,
            reason=reason,
            tags=self.tags.copy(),
        )


class SilenceTerminatedBuffer(BufferPolicy):
    """
    Silence-terminated buffer.

    Accumulates audio during speech and emits when silence is detected,
    subject to minimum and maximum duration constraints.
    """

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        min_s: float = 0.5,
        max_s: float = 10.0,
        silence_s: float = 0.5,
        gate: Optional[SpeechGate] = None,
    ):
        self.sr = sr
        self.min_samples = int(min_s * sr)
        self.max_samples = int(max_s * sr)
        self.silence_samples = int(silence_s * sr)
        self.gate = gate

        self.ring = RingBuffer(self.max_samples + sr)
        self.active = False
        self.seg_start_sample = 0
        self.seg_samples = 0
        self.trailing_silence = 0

        self.tags = {
            "min_s": min_s,
            "max_s": max_s,
            "silence_s": silence_s,
        }

    def push(self, frame_audio: np.ndarray, t0: int, t1: int) -> List[Segment]:
        segments = []

        is_speech = True
        if self.gate is not None:
            is_speech = self.gate.is_speech(frame_audio)

        if is_speech:
            if not self.active:
                self.active = True
                self.seg_start_sample = t0
                self.seg_samples = 0
                self.trailing_silence = 0

            self.ring.append(frame_audio)
            self.seg_samples += len(frame_audio)
            self.trailing_silence = 0
        else:
            if self.active:
                self.ring.append(frame_audio)
                self.seg_samples += len(frame_audio)
                self.trailing_silence += len(frame_audio)

        should_emit = False
        reason = ""

        if self.active:
            if self.seg_samples >= self.max_samples:
                should_emit = True
                reason = "max_duration"
            elif self.trailing_silence >= self.silence_samples and self.seg_samples >= self.min_samples:
                should_emit = True
                reason = "silence_end"

        if should_emit:
            seg = self._emit(t1, reason=reason)
            segments.append(seg)
            self.ring.clear()
            self.seg_samples = 0
            self.trailing_silence = 0
            self.active = False

        return segments

    def flush(self, t_end: int) -> List[Segment]:
        if self.seg_samples == 0:
            return []

        seg = self._emit(t_end, reason="flush")
        self.ring.clear()
        self.seg_samples = 0
        self.trailing_silence = 0
        self.active = False
        return [seg]

    def reset(self) -> None:
        self.ring.clear()
        self.active = False
        self.seg_start_sample = 0
        self.seg_samples = 0
        self.trailing_silence = 0

    def _emit(self, t_end: int, reason: str) -> Segment:
        audio = self.ring.get_all()
        return Segment(
            audio=audio,
            start_sample=self.seg_start_sample,
            end_sample=t_end,
            reason=reason,
            tags=self.tags.copy(),
        )


class VADBuffer(BufferPolicy):
    """
    VAD-based segmenter using frame-level speech decisions.

    Assumptions:
    - `push()` is called with fixed-hop frames; gate decision is per frame.
    - `pre_s` is real-audio pre-roll: last `pre_s` seconds before speech start.
    - `post_s` is real-audio post-roll after last speech frame (no zero padding).
    - `min_speech_s` is measured from speech-only frames, not total segment length.

    Behavior:
    - On speech start, prepends rolling pre-buffer and starts an active segment.
    - While active, appends every frame for continuity and tracks speech-only samples.
    - Emits once trailing non-speech reaches `end_silence_s` and speech-only duration
      is at least `min_speech_s`; false starts below `min_speech_s` are discarded.
    """

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        pre_s: float = 0.2,
        post_s: float = 0.2,
        min_speech_s: float = 0.4,
        end_silence_s: float = 0.5,
        max_s: float = 10.0,
        gate: Optional[SpeechGate] = None,
    ):
        self.sr = sr
        self.pre_samples = int(pre_s * sr)
        self.post_samples = int(post_s * sr)
        self.min_speech_samples = int(min_speech_s * sr)
        self.end_silence_samples = int(end_silence_s * sr)
        self.max_samples = int(max_s * sr)
        self.gate = gate

        self.ring = RingBuffer(self.max_samples + self.pre_samples + self.post_samples)
        self.pre_ring = RingBuffer(max(1, self.pre_samples) if self.pre_samples > 0 else 1)
        self.active = False
        self.seg_start_sample = 0
        self.seg_samples = 0
        self.speech_samples = 0
        self.trailing_silence = 0
        self.speech_end_seg_samples = 0

        self.tags = {
            "pre_s": pre_s,
            "post_s": post_s,
            "min_speech_s": min_speech_s,
            "end_silence_s": end_silence_s,
            "max_s": max_s,
        }

    def push(self, frame_audio: np.ndarray, t0: int, t1: int) -> List[Segment]:
        segments = []
        frame = np.asarray(frame_audio, dtype=np.float32).reshape(-1)

        is_speech = True
        if self.gate is not None:
            is_speech = self.gate.is_speech(frame)

        if is_speech:
            if not self.active:
                self.active = True
                pre_audio = (
                    self.pre_ring.slice_last(min(self.pre_samples, self.pre_ring.get_size()))
                    if self.pre_samples > 0
                    else np.zeros(0, dtype=np.float32)
                )
                pre_len = int(len(pre_audio))
                self.seg_start_sample = max(0, t0 - pre_len)
                self.seg_samples = 0
                self.speech_samples = 0
                self.trailing_silence = 0
                self.speech_end_seg_samples = 0
                if pre_len > 0:
                    self.ring.append(pre_audio)
                    self.seg_samples += pre_len

            self.ring.append(frame)
            self.seg_samples += len(frame)
            self.speech_samples += len(frame)
            self.speech_end_seg_samples = self.seg_samples
            self.trailing_silence = 0
        else:
            if self.active:
                self.ring.append(frame)
                self.seg_samples += len(frame)
                self.trailing_silence += len(frame)

        should_emit = False
        should_discard = False
        reason = ""

        if self.active:
            if self.seg_samples >= self.max_samples:
                if self.speech_samples >= self.min_speech_samples:
                    should_emit = True
                    reason = "max_duration"
                else:
                    should_discard = True
            elif self.trailing_silence >= self.end_silence_samples:
                if self.speech_samples >= self.min_speech_samples:
                    should_emit = True
                    reason = "silence_end"
                else:
                    should_discard = True

        if should_emit:
            trim_to = self.seg_samples
            if reason == "silence_end":
                trim_to = min(self.seg_samples, self.speech_end_seg_samples + self.post_samples)
            seg = self._emit(t1, reason=reason, trim_to_samples=trim_to)
            segments.append(seg)
            self._reset_active_segment()
        elif should_discard:
            self._reset_active_segment()

        if self.pre_samples > 0:
            self.pre_ring.append(frame)

        return segments

    def flush(self, t_end: int) -> List[Segment]:
        if self.seg_samples == 0:
            return []

        if self.speech_samples < self.min_speech_samples:
            self._reset_active_segment()
            return []

        trim_to = min(self.seg_samples, self.speech_end_seg_samples + self.post_samples)
        seg = self._emit(t_end, reason="flush", trim_to_samples=trim_to)
        self._reset_active_segment()
        return [seg]

    def reset(self) -> None:
        self.ring.clear()
        self.pre_ring.clear()
        self.active = False
        self.seg_start_sample = 0
        self.seg_samples = 0
        self.speech_samples = 0
        self.trailing_silence = 0
        self.speech_end_seg_samples = 0

    def _reset_active_segment(self) -> None:
        self.ring.clear()
        self.active = False
        self.seg_start_sample = 0
        self.seg_samples = 0
        self.speech_samples = 0
        self.trailing_silence = 0
        self.speech_end_seg_samples = 0

    def _emit(self, t_end: int, reason: str, trim_to_samples: Optional[int] = None) -> Segment:
        audio = self.ring.get_all().astype(np.float32, copy=False)
        if trim_to_samples is not None:
            trim_to = max(0, int(trim_to_samples))
            if trim_to < len(audio):
                audio = audio[:trim_to]

        end_sample = self.seg_start_sample + len(audio)
        return Segment(
            audio=audio,
            start_sample=self.seg_start_sample,
            end_sample=end_sample if reason == "silence_end" else t_end,
            reason=reason,
            tags=self.tags.copy(),
        )


if __name__ == "__main__":
    class _SeqGate:
        def __init__(self, seq: list[bool]):
            self.seq = seq
            self.idx = 0

        def is_speech(self, _frame: np.ndarray) -> bool:
            if self.idx >= len(self.seq):
                return False
            out = self.seq[self.idx]
            self.idx += 1
            return out

    def _run_sim(sequence: list[bool], values: list[float], sr: int = 16000, hop_s: float = 0.1) -> list[Segment]:
        hop = int(sr * hop_s)
        gate = _SeqGate(sequence)
        buf = VADBuffer(
            sr=sr,
            pre_s=0.2,
            post_s=0.2,
            min_speech_s=0.4,
            end_silence_s=0.5,
            max_s=30.0,
            gate=gate,
        )
        out: list[Segment] = []
        t0 = 0
        for is_speech, val in zip(sequence, values):
            frame = np.full(hop, val, dtype=np.float32)
            t1 = t0 + hop
            out.extend(buf.push(frame, t0=t0, t1=t1))
            t0 = t1
        out.extend(buf.flush(t_end=t0))
        return out

    # 1) 0.3s speech + silence: below min_speech_s=0.4 -> no emission.
    seq1 = [False, False] + [True] * 3 + [False] * 8
    vals1 = [0.7, 0.6] + [1.0] * 3 + [0.0] * 8
    segs1 = _run_sim(seq1, vals1)
    assert len(segs1) == 0, f"Expected no segment for short speech, got {len(segs1)}"

    # 2) 0.6s speech + 0.6s silence: emit with pre(0.2s)+post(0.2s).
    seq2 = [False, False] + [True] * 6 + [False] * 6
    vals2 = [0.7, 0.6] + [1.0] * 6 + [0.0] * 6
    segs2 = _run_sim(seq2, vals2)
    assert len(segs2) == 1, f"Expected one segment, got {len(segs2)}"
    seg = segs2[0]
    expected_len = int((0.2 + 0.6 + 0.2) * 16000)
    tol = int(0.1 * 16000)
    assert abs(len(seg.audio) - expected_len) <= tol, (
        f"Segment length mismatch. got={len(seg.audio)} expected≈{expected_len}"
    )

    # 3) Pre-roll should be real audio (non-zero), not synthetic zeros.
    assert np.abs(seg.audio[0]) > 1e-6, "Leading sample is zero; pre-roll should contain real preceding audio"

    print("VADBuffer self-check passed.")
