"""
ASR commit policies for the streaming pipeline.

Determines which parts of ASR hypotheses are stable (committed) vs revisable.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..core.types import ASROutput, ASRCommitState, DEFAULT_MIN_STABLE_STEPS
from ..buffers.merge import MergeConfig, merge_transcripts


class ASRCommitPolicy(ABC):
    """Abstract interface for ASR commitment strategies."""

    @abstractmethod
    def update(self, hyp: ASROutput) -> ASRCommitState:
        """Update state with new hypothesis."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        """Reset to initial state."""
        raise NotImplementedError()


class StabilityASRCommit(ASRCommitPolicy):
    """Stability-based ASR commitment."""

    def __init__(self, min_stable_steps: int = DEFAULT_MIN_STABLE_STEPS):
        self.min_stable_steps = min_stable_steps
        self.history: List[str] = []
        self.committed = ""
        self.pending = ""
        self._last_delta = ""

    def update(self, hyp: ASROutput) -> ASRCommitState:
        current_text = hyp.text.strip()
        self.history.append(current_text)

        if len(self.history) > self.min_stable_steps + 1:
            self.history = self.history[-(self.min_stable_steps + 1):]

        if len(self.history) >= self.min_stable_steps:
            stable_prefix = self._stable_prefix(self.history[-self.min_stable_steps:])

            if len(stable_prefix) > len(self.committed):
                new_committed = stable_prefix
                self._last_delta = new_committed[len(self.committed):]
                self.committed = new_committed
            else:
                self._last_delta = ""
        else:
            self._last_delta = ""

        if current_text.startswith(self.committed):
            self.pending = current_text[len(self.committed):]
        else:
            self.pending = current_text

        return ASRCommitState(
            committed=self.committed,
            pending=self.pending,
            delta_committed=self._last_delta,
        )

    def reset(self) -> None:
        self.history.clear()
        self.committed = ""
        self.pending = ""
        self._last_delta = ""

    def _stable_prefix(self, texts: List[str]) -> str:
        if not texts:
            return ""

        prefix = texts[0]
        for text in texts[1:]:
            prefix = self._lcp(prefix, text)
            if not prefix:
                break

        return self._trim_to_word_boundary(prefix)

    def _lcp(self, a: str, b: str) -> str:
        min_len = min(len(a), len(b))
        for i in range(min_len):
            if a[i] != b[i]:
                return a[:i]
        return a[:min_len]

    def _trim_to_word_boundary(self, text: str) -> str:
        if not text:
            return ""
        if text.endswith(" "):
            return text
        last_space = text.rfind(" ")
        if last_space > 0:
            return text[:last_space + 1]
        return text


class ImmediateASRCommit(ASRCommitPolicy):
    """Immediate ASR commitment - commits everything immediately."""

    def __init__(self):
        self.committed = ""
        self._last_delta = ""

    def update(self, hyp: ASROutput) -> ASRCommitState:
        current_text = hyp.text.strip()

        if current_text.startswith(self.committed):
            self._last_delta = current_text[len(self.committed):]
        else:
            self._last_delta = current_text

        self.committed = current_text

        return ASRCommitState(
            committed=self.committed,
            pending="",
            delta_committed=self._last_delta,
        )

    def reset(self) -> None:
        self.committed = ""
        self._last_delta = ""


class WindowedASRCommit(ASRCommitPolicy):
    """Windowed ASR commitment - commits older words, keeps recent pending."""

    def __init__(self, commit_words_behind: int = 5):
        self.commit_words_behind = commit_words_behind
        self.committed = ""
        self._last_delta = ""

    def update(self, hyp: ASROutput) -> ASRCommitState:
        current_text = hyp.text.strip()
        words = current_text.split()

        pending = current_text

        if len(words) > self.commit_words_behind:
            commit_words = words[:-self.commit_words_behind]
            pending_words = words[-self.commit_words_behind:]

            new_committed = " ".join(commit_words) + " "
            pending = " ".join(pending_words)

            if len(new_committed) > len(self.committed):
                self._last_delta = new_committed[len(self.committed):]
                self.committed = new_committed
            else:
                self._last_delta = ""
        else:
            self._last_delta = ""

        return ASRCommitState(
            committed=self.committed,
            pending=pending,
            delta_committed=self._last_delta,
        )

    def reset(self) -> None:
        self.committed = ""
        self._last_delta = ""


class MergingASRCommit(ASRCommitPolicy):
    """
    ASR commit policy with transcript merging for overlapping windows.
    """

    def __init__(
        self,
        merge_cfg: Optional[MergeConfig] = None,
        commit_immediately: bool = True,
    ):
        self.merge_cfg = merge_cfg or MergeConfig()
        self.commit_immediately = commit_immediately
        self.merged_text = ""
        self._last_delta = ""

    def update(self, hyp: ASROutput) -> ASRCommitState:
        current_text = hyp.text.strip()

        if not current_text:
            return ASRCommitState(
                committed=self.merged_text,
                pending="",
                delta_committed="",
            )

        prev_len = len(self.merged_text)
        self.merged_text = merge_transcripts(self.merged_text, current_text, self.merge_cfg)

        if len(self.merged_text) > prev_len:
            self._last_delta = self.merged_text[prev_len:].lstrip()
        else:
            self._last_delta = ""

        return ASRCommitState(
            committed=self.merged_text,
            pending="",
            delta_committed=self._last_delta,
        )

    def reset(self) -> None:
        self.merged_text = ""
        self._last_delta = ""
