"""Model manager for ASR and MT loading."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from streaming.asr.asr import ASRModel, create_asr_model
from streaming.mt.mt import MTModel, create_mt_model

from .session import SessionConfig


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelManagerStatus:
    """Runtime snapshot of model loading state."""

    asr_loaded: bool
    mt_loaded: bool
    asr_model_id: str
    mt_model_id: str
    device: str
    offline: bool


class ModelManager:
    """Load and hold ASR/MT model instances for a session."""

    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        self.asr: Optional[ASRModel] = None
        self.mt: Optional[MTModel] = None

    def load_asr(self) -> ASRModel:
        """Load ASR model lazily and return it."""
        if self.asr is None:
            self.asr = self._create_asr_model()
        return self.asr

    def load_mt(self) -> MTModel:
        """Load MT model lazily and return it."""
        if self.mt is None:
            self.mt = self._create_mt_model()
        return self.mt

    def load_all(self) -> tuple[ASRModel, MTModel]:
        """Load both ASR and MT models."""
        return self.load_asr(), self.load_mt()

    def warmup(self) -> None:
        """Trigger lazy loading of all models."""
        self.load_all()

    def status(self) -> ModelManagerStatus:
        """Return a lightweight status object for logs/health checks."""
        return ModelManagerStatus(
            asr_loaded=self.asr is not None,
            mt_loaded=self.mt is not None,
            asr_model_id=self.cfg.models.asr_model_id,
            mt_model_id=self.cfg.models.mt_model_id,
            device=self.cfg.runtime.device,
            offline=self.cfg.runtime.offline,
        )

    def _create_asr_model(self) -> ASRModel:
        device = self._resolve_device(self.cfg.runtime.device)
        try:
            return create_asr_model(
                model_id=self.cfg.models.asr_model_id,
                device=device,
                language=self.cfg.models.asr_language,
                offline=self.cfg.runtime.offline,
                cache_dir=self.cfg.runtime.cache_dir,
            )
        except Exception as exc:
            _LOGGER.warning("Falling back to dummy ASR model: %s", exc)
            return create_asr_model(model_id="dummy")

    def _create_mt_model(self) -> MTModel:
        device = self._resolve_device(self.cfg.runtime.device)
        src_lang = (self.cfg.models.asr_language or "").strip().lower() or None
        tgt_lang = self._infer_mt_target_lang(self.cfg.models.mt_model_id)
        try:
            return create_mt_model(
                model_id=self.cfg.models.mt_model_id,
                device=device,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                offline=self.cfg.runtime.offline,
                cache_dir=self.cfg.runtime.cache_dir,
            )
        except Exception as exc:
            _LOGGER.warning("Falling back to dummy MT model: %s", exc)
            return create_mt_model(model_id="dummy")

    @staticmethod
    def _infer_mt_target_lang(model_id: str) -> Optional[str]:
        model_key = (model_id or "").strip().lower()
        if not model_key:
            return None

        pair_match = re.search(r"(?:^|[-_/])([a-z]{2,3})-([a-z]{2,3})(?:$|[-_/])", model_key)
        if pair_match:
            return pair_match.group(2)

        if "m2m100" in model_key:
            return "en"

        return None

    @staticmethod
    def _resolve_device(device: str) -> Optional[str]:
        normalized = (device or "").strip().lower()
        if normalized in {"", "auto"}:
            return None
        return device
