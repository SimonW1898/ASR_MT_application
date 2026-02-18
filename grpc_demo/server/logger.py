"""JSONL logger for gRPC demo sessions."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

from .processor import ProcessorResult


class JsonlLogger:
    """JSONL logger for segment results."""

    def __init__(self, log_dir: str, session_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = session_id or f"grpc_demo_{ts}"
        self.path = self.log_dir / f"{name}.jsonl"
        self.fh = open(self.path, "w", encoding="utf-8")

    def write_result(self, result: ProcessorResult) -> None:
        record = {
            "segment_id": result.segment_id,
            "asr_chunk_text": result.asr_chunk_text,
            "mt_chunk_text": result.mt_chunk_text,
            "merged_asr": result.merged_asr,
            "merged_mt": result.merged_mt,
            "reason": result.reason,
            "audio_time_s": result.audio_time_s,
            "queue_latency_ms": result.queue_latency_ms,
            "process_time_ms": result.process_time_ms,
            "e2e_ms": result.e2e_ms,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()
