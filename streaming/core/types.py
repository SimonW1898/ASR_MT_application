"""
Core data types for the streaming ASR to MT pipeline.

All timing is in samples at a known sampling rate (default 16 kHz).
Audio arrays are mono float32 in [-1, 1].
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class Frame:
    """A single chunk from an audio source."""
    audio: np.ndarray  # float32, shape (n,)
    t0: int  # start sample index (global)
    t1: int  # end sample index (global)
    frame_idx: int  # sequential frame counter
    meta: Optional[Dict[str, Any]] = None  # e.g., device id, file name


@dataclass
class Segment:
    """An audio chunk emitted by a buffer policy for ASR."""
    audio: np.ndarray  # float32
    start_sample: int  # global sample index
    end_sample: int  # global sample index
    reason: str  # e.g., "waitk_hop", "silence_end", "fixed_window", "flush"
    tags: Optional[Dict[str, Any]] = None  # debugging info


@dataclass
class ASROutput:
    """Result of a single ASR call."""
    text: str
    extra: Optional[Dict[str, Any]] = None  # timestamps, logprobs, etc.


@dataclass
class ASRCommitState:
    """Stable ASR state derived from hypotheses."""
    committed: str  # stable text (monotone growth)
    pending: str  # revisable tail
    delta_committed: str  # newly committed text since last update

    @property
    def full_text(self) -> str:
        """Full hypothesis: committed + pending."""
        return self.committed + self.pending


@dataclass
class MTScope:
    """Decision describing what to translate at this step."""
    text_to_translate: str
    scope_id: str  # identifier, e.g., "delta", "delta_plus_tail"
    is_incremental: bool  # True if translation is append-only


@dataclass
class MTCommitState:
    """Stable MT state."""
    committed: str  # stable translation
    pending: str  # revisable tail
    delta_committed: str  # newly committed since last update

    @property
    def full_text(self) -> str:
        """Full translation: committed + pending."""
        return self.committed + self.pending


# Default constants
DEFAULT_SR = 16000
DEFAULT_FRAME_MS = 100
DEFAULT_FRAME_SAMPLES = int(DEFAULT_SR * DEFAULT_FRAME_MS / 1000)  # 1600

# Wait-k defaults
DEFAULT_K_S = 2.0  # seconds
DEFAULT_HOP_S = 1.0  # seconds
DEFAULT_WIN_S = 6.0  # seconds

# ASR commit defaults
DEFAULT_MIN_STABLE_STEPS = 2

# MT scope defaults
DEFAULT_TAIL_CHARS = 200
