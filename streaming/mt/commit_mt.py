"""
MT commit policies for the streaming pipeline.

Determines how to handle MT hypotheses and maintain stable translations.
"""

from abc import ABC, abstractmethod

from ..core.types import MTScope, MTCommitState


class MTCommitPolicy(ABC):
    """Abstract interface for MT commitment strategies."""

    @abstractmethod
    def update(self, scope: MTScope, mt_hyp: str) -> MTCommitState:
        """Update state with new MT hypothesis."""
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        """Reset to initial state."""
        raise NotImplementedError()


class ReplacePendingMT(MTCommitPolicy):
    """Replace pending MT policy."""

    def __init__(self):
        self.committed = ""
        self.pending = ""
        self._last_delta = ""

    def update(self, scope: MTScope, mt_hyp: str) -> MTCommitState:
        mt_hyp = mt_hyp.strip()

        if not mt_hyp:
            return MTCommitState(committed=self.committed, pending=self.pending, delta_committed="")

        if scope.is_incremental:
            self._last_delta = mt_hyp
            self.committed = self.committed + (" " if self.committed else "") + mt_hyp
            self.pending = ""
        else:
            self.pending = mt_hyp
            self._last_delta = ""

        return MTCommitState(committed=self.committed, pending=self.pending, delta_committed=self._last_delta)

    def reset(self) -> None:
        self.committed = ""
        self.pending = ""
        self._last_delta = ""


class AppendOnlyMT(MTCommitPolicy):
    """Append-only MT policy."""

    def __init__(self):
        self.committed = ""
        self._last_delta = ""

    def update(self, scope: MTScope, mt_hyp: str) -> MTCommitState:
        mt_hyp = mt_hyp.strip()

        if not mt_hyp:
            return MTCommitState(committed=self.committed, pending="", delta_committed="")

        self._last_delta = mt_hyp
        if self.committed:
            self.committed = self.committed + " " + mt_hyp
        else:
            self.committed = mt_hyp

        return MTCommitState(committed=self.committed, pending="", delta_committed=self._last_delta)

    def reset(self) -> None:
        self.committed = ""
        self._last_delta = ""


class FullReplaceMT(MTCommitPolicy):
    """Full replace MT policy."""

    def __init__(self):
        self.current = ""

    def update(self, scope: MTScope, mt_hyp: str) -> MTCommitState:
        mt_hyp = mt_hyp.strip()
        delta = mt_hyp if mt_hyp != self.current else ""
        self.current = mt_hyp
        return MTCommitState(committed=self.current, pending="", delta_committed=delta)

    def reset(self) -> None:
        self.current = ""


class StabilityMT(MTCommitPolicy):
    """Stability-based MT policy."""

    def __init__(self, min_stable_steps: int = 2):
        self.min_stable_steps = min_stable_steps
        self.history = []
        self.committed = ""
        self.pending = ""
        self._last_delta = ""

    def update(self, scope: MTScope, mt_hyp: str) -> MTCommitState:
        mt_hyp = mt_hyp.strip()

        self.history.append(mt_hyp)
        if len(self.history) > self.min_stable_steps + 1:
            self.history = self.history[-(self.min_stable_steps + 1):]

        if len(self.history) >= self.min_stable_steps:
            stable_prefix = self._stable_prefix(self.history[-self.min_stable_steps:])

            if len(stable_prefix) > len(self.committed):
                self._last_delta = stable_prefix[len(self.committed):]
                self.committed = stable_prefix
            else:
                self._last_delta = ""
        else:
            self._last_delta = ""

        if mt_hyp.startswith(self.committed):
            self.pending = mt_hyp[len(self.committed):]
        else:
            self.pending = mt_hyp

        return MTCommitState(committed=self.committed, pending=self.pending, delta_committed=self._last_delta)

    def reset(self) -> None:
        self.history.clear()
        self.committed = ""
        self.pending = ""
        self._last_delta = ""

    def _stable_prefix(self, texts):
        if not texts:
            return ""
        prefix = texts[0]
        for text in texts[1:]:
            prefix = self._lcp(prefix, text)
            if not prefix:
                break
        return self._trim_to_word(prefix)

    def _lcp(self, a, b):
        min_len = min(len(a), len(b))
        for i in range(min_len):
            if a[i] != b[i]:
                return a[:i]
        return a[:min_len]

    def _trim_to_word(self, text):
        if not text or text.endswith(" "):
            return text
        last_space = text.rfind(" ")
        return text[:last_space + 1] if last_space > 0 else text
