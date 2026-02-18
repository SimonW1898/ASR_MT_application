"""Translation processor (ASR -> merge -> MT)."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from streaming.buffers.merge import AlignMergeConfig, merge_transcripts_align
from streaming.core.types import Segment

from .model_manager import ModelManager
from .session import SessionConfig


@dataclass
class ProcessorStats:
    segments_processed: int = 0


@dataclass
class QueuedSegment:
    segment: Segment
    enqueue_time: float


@dataclass
class ProcessorResult:
    segment_id: int
    asr_chunk_text: str
    mt_chunk_text: str
    merged_asr: str
    merged_mt: str
    reason: str
    audio_time_s: float
    queue_latency_ms: float
    process_time_ms: float
    e2e_ms: Optional[float]


class TranslationProcessor:
    """Processor that runs ASR -> merge -> MT for emitted segments."""

    def __init__(
        self,
        cfg: SessionConfig,
        result_queue: queue.Queue,
        model_manager: Optional[ModelManager] = None,
    ):
        self.cfg = cfg
        self.result_queue = result_queue
        self.stats = ProcessorStats()
        self._models = model_manager or ModelManager(cfg)

        self._in_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()

        self._merge_cfg = AlignMergeConfig(
            tail_tokens=cfg.merge.tail_tokens,
            head_tokens=cfg.merge.head_tokens,
            min_match_tokens=cfg.merge.min_match_tokens,
            max_search_shift=cfg.merge.max_search_shift,
        )
        self._merged_asr = ""
        self._merged_mt = ""

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._in_queue.put(None)
        if self._thread:
            self._thread.join(timeout=5.0)

    def is_stopped(self) -> bool:
        return self._stopped.is_set()

    def submit(self, segment: Segment, enqueue_time: float) -> None:
        self._in_queue.put(QueuedSegment(segment=segment, enqueue_time=enqueue_time))

    def _run(self) -> None:
        while True:
            item = self._in_queue.get()
            if item is None:
                self._in_queue.task_done()
                break

            result = self._process(item)
            self.result_queue.put(result)
            self._in_queue.task_done()

        self._stopped.set()

    def _process(self, item: QueuedSegment) -> ProcessorResult:
        process_start = time.time()

        segment_id = self.stats.segments_processed + 1
        asr = self._models.load_asr()
        mt = self._models.load_mt()

        asr_output = asr.transcribe(item.segment.audio, sr=self.cfg.audio.sample_rate)
        asr_text = (asr_output.text or "").strip()

        if self.cfg.merge.enabled:
            if self._merged_asr:
                self._merged_asr = merge_transcripts_align(
                    self._merged_asr,
                    asr_text,
                    self._merge_cfg,
                )
            else:
                self._merged_asr = asr_text
        else:
            if self._merged_asr:
                self._merged_asr = self._merged_asr + " " + asr_text
            else:
                self._merged_asr = asr_text

        if self.cfg.mt.mode == "merged":
            mt_input = self._merged_asr
            mt_text = mt.translate(mt_input) if mt_input else ""
            self._merged_mt = mt_text
        else:
            mt_text = mt.translate(asr_text) if asr_text else ""
            if mt_text and self._merged_mt:
                self._merged_mt = self._merged_mt + " " + mt_text
            elif mt_text:
                self._merged_mt = mt_text

        now = time.time()
        process_ms = (now - process_start) * 1000.0
        queue_ms = (process_start - item.enqueue_time) * 1000.0
        e2e_ms = queue_ms + process_ms

        self.stats.segments_processed = segment_id

        return ProcessorResult(
            segment_id=segment_id,
            asr_chunk_text=asr_text,
            mt_chunk_text=mt_text,
            merged_asr=self._merged_asr,
            merged_mt=self._merged_mt,
            reason=item.segment.reason,
            audio_time_s=item.segment.end_sample / self.cfg.audio.sample_rate,
            queue_latency_ms=queue_ms,
            process_time_ms=process_ms,
            e2e_ms=e2e_ms,
        )
