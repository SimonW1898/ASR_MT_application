"""
MT scope policies for the streaming pipeline.

Determines what portion of ASR output to translate at each step.
"""

from abc import ABC, abstractmethod

from ..core.types import ASRCommitState, MTScope, DEFAULT_TAIL_CHARS


class MTScopePolicy(ABC):
    """Abstract interface for MT scope selection."""

    @abstractmethod
    def select(self, asr_state: ASRCommitState) -> MTScope:
        """Determine what text to translate."""
        raise NotImplementedError()


class TranslateCommittedOnly(MTScopePolicy):
    """Translate only committed (stable) ASR text."""

    def __init__(self):
        self._last_committed = ""

    def select(self, asr_state: ASRCommitState) -> MTScope:
        if asr_state.delta_committed:
            text = asr_state.committed
            self._last_committed = text
            return MTScope(text_to_translate=text, scope_id="committed_only", is_incremental=True)
        return MTScope(text_to_translate="", scope_id="committed_only", is_incremental=True)


class TranslateCommittedPlusTail(MTScopePolicy):
    """Translate committed text plus a tail of pending text."""

    def __init__(self, tail_chars: int = DEFAULT_TAIL_CHARS):
        self.tail_chars = tail_chars

    def select(self, asr_state: ASRCommitState) -> MTScope:
        committed = asr_state.committed
        pending = asr_state.pending
        tail = pending[:self.tail_chars] if len(pending) > self.tail_chars else pending
        text_to_translate = committed + tail

        if not text_to_translate.strip():
            return MTScope(text_to_translate="", scope_id="committed_plus_tail", is_incremental=False)

        return MTScope(text_to_translate=text_to_translate, scope_id="committed_plus_tail", is_incremental=False)


class TranslateFull(MTScopePolicy):
    """Translate full ASR hypothesis (committed + all pending)."""

    def select(self, asr_state: ASRCommitState) -> MTScope:
        full_text = asr_state.full_text

        if not full_text.strip():
            return MTScope(text_to_translate="", scope_id="full", is_incremental=False)

        return MTScope(text_to_translate=full_text, scope_id="full", is_incremental=False)


class TranslateDeltaOnly(MTScopePolicy):
    """Translate only the newly committed delta."""

    def select(self, asr_state: ASRCommitState) -> MTScope:
        delta = asr_state.delta_committed

        if not delta.strip():
            return MTScope(text_to_translate="", scope_id="delta_only", is_incremental=True)

        return MTScope(text_to_translate=delta, scope_id="delta_only", is_incremental=True)
