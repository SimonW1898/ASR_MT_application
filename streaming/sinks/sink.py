"""
Event sinks for the streaming pipeline.

Handles output of pipeline events for logging, evaluation, and debugging.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, List
import json
import datetime

import numpy as np

from ..core.types import Segment, ASROutput, ASRCommitState, MTScope, MTCommitState, DEFAULT_SR


class EventSink(ABC):
    """Abstract interface for event sinks."""

    @abstractmethod
    def on_segment(
        self,
        segment: Segment,
        asr_out: ASROutput,
        asr_state: ASRCommitState,
        scope: MTScope,
        mt_hyp: str,
        mt_state: MTCommitState,
    ) -> None:
        """Handle a processed segment event."""
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Finalize and release resources."""
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ConsoleSink(EventSink):
    """Print events to console."""

    def __init__(self, show_pending: bool = True, show_timing: bool = True):
        self.show_pending = show_pending
        self.show_timing = show_timing
        self.event_count = 0

    def on_segment(
        self,
        segment: Segment,
        asr_out: ASROutput,
        asr_state: ASRCommitState,
        scope: MTScope,
        mt_hyp: str,
        mt_state: MTCommitState,
    ) -> None:
        self.event_count += 1

        if self.show_timing:
            start_s = segment.start_sample / DEFAULT_SR
            end_s = segment.end_sample / DEFAULT_SR
            print(f"\n[{self.event_count}] {start_s:.2f}s - {end_s:.2f}s ({segment.reason})")
        else:
            print(f"\n[{self.event_count}] ({segment.reason})")

        if self.show_pending:
            print(f"  ASR: {asr_state.committed}|{asr_state.pending}")
        else:
            print(f"  ASR: {asr_state.committed}")

        if self.show_pending:
            print(f"  MT:  {mt_state.committed}|{mt_state.pending}")
        else:
            print(f"  MT:  {mt_state.committed}")

    def close(self) -> None:
        print(f"\n--- Processed {self.event_count} segments ---")


class JsonlSink(EventSink):
    """Write events to JSONL file."""

    def __init__(self, path: Path, pretty: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w", encoding="utf-8")
        self.pretty = pretty
        self.event_count = 0

    def on_segment(
        self,
        segment: Segment,
        asr_out: ASROutput,
        asr_state: ASRCommitState,
        scope: MTScope,
        mt_hyp: str,
        mt_state: MTCommitState,
    ) -> None:
        self.event_count += 1

        record = {
            "event_idx": self.event_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "segment": {
                "start_sample": segment.start_sample,
                "end_sample": segment.end_sample,
                "start_s": segment.start_sample / DEFAULT_SR,
                "end_s": segment.end_sample / DEFAULT_SR,
                "reason": segment.reason,
                "tags": segment.tags,
            },
            "asr": {
                "raw": asr_out.text,
                "committed": asr_state.committed,
                "pending": asr_state.pending,
                "delta": asr_state.delta_committed,
            },
            "mt_scope": {
                "text": scope.text_to_translate,
                "scope_id": scope.scope_id,
                "is_incremental": scope.is_incremental,
            },
            "mt": {
                "raw": mt_hyp,
                "committed": mt_state.committed,
                "pending": mt_state.pending,
                "delta": mt_state.delta_committed,
            },
        }

        if self.pretty:
            line = json.dumps(record, indent=2, ensure_ascii=False)
        else:
            line = json.dumps(record, ensure_ascii=False)

        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class SegmentWavSink(EventSink):
    """Save audio segments as WAV files."""

    def __init__(self, out_dir: Path, sr: int = DEFAULT_SR):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sr
        self.counter = 0

    def on_segment(
        self,
        segment: Segment,
        asr_out: ASROutput,
        asr_state: ASRCommitState,
        scope: MTScope,
        mt_hyp: str,
        mt_state: MTCommitState,
    ) -> None:
        import scipy.io.wavfile as wav

        self.counter += 1
        filename = f"segment_{self.counter:04d}_{segment.reason}.wav"
        filepath = self.out_dir / filename

        audio_int16 = (segment.audio * 32767).astype(np.int16)
        wav.write(str(filepath), self.sr, audio_int16)

    def close(self) -> None:
        pass


class MultiSink(EventSink):
    """Multiplex events to multiple sinks."""

    def __init__(self, sinks: List[EventSink]):
        self.sinks = sinks

    def on_segment(
        self,
        segment: Segment,
        asr_out: ASROutput,
        asr_state: ASRCommitState,
        scope: MTScope,
        mt_hyp: str,
        mt_state: MTCommitState,
    ) -> None:
        for sink in self.sinks:
            sink.on_segment(segment, asr_out, asr_state, scope, mt_hyp, mt_state)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


class NullSink(EventSink):
    """Discard all events."""

    def on_segment(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class RealtimeSink(ABC):
    """Abstract interface for real-time translation output."""

    @abstractmethod
    def on_result(self, result: Any) -> None:
        """Handle a translation result."""
        raise NotImplementedError()

    @abstractmethod
    def on_final(self, merged_asr: str, merged_mt: str, timing_stats: dict) -> None:
        """Handle final output when pipeline stops."""
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Finalize and release resources."""
        raise NotImplementedError()


class RealtimeConsoleSink(RealtimeSink):
    """Console output for real-time translation."""

    def __init__(
        self,
        show_segment_text: bool = True,
        show_merged: bool = True,
        show_timing: bool = True,
        line_width: int = 70,
    ):
        self.show_segment_text = show_segment_text
        self.show_merged = show_merged
        self.show_timing = show_timing
        self.line_width = line_width
        self.segment_count = 0

    def on_result(self, result: Any) -> None:
        self.segment_count += 1

        print("\n" + "-" * self.line_width)
        print(f"[Segment {result.segment_idx}] @ {result.audio_time_s:.2f}s | Reason: {result.reason}")

        if self.show_timing:
            timing_parts = [
                f"Queue: {result.queue_latency_s*1000:.0f}ms",
                f"Process: {result.process_time_s*1000:.0f}ms",
            ]
            if result.first_word_to_translation_s is not None:
                timing_parts.append(f"First->Trans: {result.first_word_to_translation_s*1000:.0f}ms")
            print(f"  {' | '.join(timing_parts)}")

        print("-" * self.line_width)

        if self.show_segment_text:
            print(f"  ASR (this):   {result.asr_text}")
            print(f"  MT  (this):   {result.mt_text}")
            print("-" * self.line_width)

        if self.show_merged:
            print(f"  [MERGED ASR]: {result.merged_asr}")
            print(f"  [MERGED MT]:  {result.merged_mt}")
            print("-" * self.line_width)

    def on_final(self, merged_asr: str, merged_mt: str, timing_stats: dict) -> None:
        print("\n" + "=" * self.line_width)
        print("  FINAL OUTPUT")
        print("=" * self.line_width)
        print(f"\n  [FINAL ASR]:\n    {merged_asr}")
        print(f"\n  [FINAL MT]:\n    {merged_mt}")
        print("\n" + "-" * self.line_width)
        print("  TIMING STATISTICS")
        print("-" * self.line_width)
        print(f"  Segments processed: {timing_stats.get('segments_processed', 0)}")
        if timing_stats.get("first_word_to_first_translation_s") is not None:
            fwtt = timing_stats["first_word_to_first_translation_s"]
            print(f"  First speech -> First translation: {fwtt*1000:.0f}ms ({fwtt:.2f}s)")
        print("=" * self.line_width)

    def close(self) -> None:
        pass


class RealtimeJsonlSink(RealtimeSink):
    """JSONL output for real-time translation."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w", encoding="utf-8")

    def on_result(self, result: Any) -> None:
        record = {
            "segment_idx": result.segment_idx,
            "audio_time_s": result.audio_time_s,
            "reason": result.reason,
            "asr_text": result.asr_text,
            "mt_text": result.mt_text,
            "merged_asr": result.merged_asr,
            "merged_mt": result.merged_mt,
            "queue_latency_s": result.queue_latency_s,
            "process_time_s": result.process_time_s,
            "first_word_to_translation_s": result.first_word_to_translation_s,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.fh.flush()

    def on_final(self, merged_asr: str, merged_mt: str, timing_stats: dict) -> None:
        record = {
            "type": "final",
            "merged_asr": merged_asr,
            "merged_mt": merged_mt,
            "timing_stats": timing_stats,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class RealtimeMultiSink(RealtimeSink):
    """Multiplex results to multiple sinks."""

    def __init__(self, sinks: List[RealtimeSink]):
        self.sinks = sinks

    def on_result(self, result: Any) -> None:
        for sink in self.sinks:
            sink.on_result(result)

    def on_final(self, merged_asr: str, merged_mt: str, timing_stats: dict) -> None:
        for sink in self.sinks:
            sink.on_final(merged_asr, merged_mt, timing_stats)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


class RealtimeNullSink(RealtimeSink):
    """Discard all results."""

    def on_result(self, result: Any) -> None:
        pass

    def on_final(self, merged_asr: str, merged_mt: str, timing_stats: dict) -> None:
        pass

    def close(self) -> None:
        pass
