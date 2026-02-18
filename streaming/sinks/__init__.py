"""Output sinks for pipeline results."""

from .sink import (
    EventSink,
    ConsoleSink,
    JsonlSink,
    SegmentWavSink,
    MultiSink,
    NullSink,
    RealtimeSink,
    RealtimeConsoleSink,
    RealtimeJsonlSink,
    RealtimeMultiSink,
    RealtimeNullSink,
)

__all__ = [
    "EventSink",
    "ConsoleSink",
    "JsonlSink",
    "SegmentWavSink",
    "MultiSink",
    "NullSink",
    "RealtimeSink",
    "RealtimeConsoleSink",
    "RealtimeJsonlSink",
    "RealtimeMultiSink",
    "RealtimeNullSink",
]
