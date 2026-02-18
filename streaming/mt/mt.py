"""
MT model wrappers for the streaming pipeline.

Uses HuggingFace models with local caching for offline use.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Optional, Any, Tuple

import torch

from ..models import load_mt_model_and_tokenizer, HF_SNAPSHOT_ROOT


class MTModel(ABC):
    """Abstract interface for machine translation models."""

    @abstractmethod
    def translate(self, text: str) -> str:
        """Translate text from source to target language."""
        raise NotImplementedError()


@lru_cache(maxsize=8)
def _get_mt_model_and_tokenizer(
    model_id: str,
    offline: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Get or create cached MT model and tokenizer from local snapshot."""
    cache_path = Path(cache_dir) if cache_dir else HF_SNAPSHOT_ROOT

    print(f"[MT] Loading model for {model_id} (offline={offline})")
    return load_mt_model_and_tokenizer(
        model_id,
        offline=offline,
        cache_dir=cache_path,
    )


class PipelineMT(MTModel):
    """Generic MT using HuggingFace models with local caching."""

    def __init__(
        self,
        model_id: str = "Helsinki-NLP/opus-mt-en-de",
        device: Optional[str] = None,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        max_length: int = 512,
        offline: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model, self.tokenizer = _get_mt_model_and_tokenizer(
            model_id, offline, cache_dir
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        if src_lang and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_lang

        print(f"[MT] Model ready: {model_id}")

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        try:
            inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            generate_kwargs = {"max_new_tokens": self.max_length}

            if self.tgt_lang and hasattr(self.tokenizer, "lang_code_to_id"):
                generate_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id.get(
                    self.tgt_lang, self.tokenizer.lang_code_to_id.get(f"{self.tgt_lang}_XX")
                )

            output_ids = self.model.generate(**inputs, **generate_kwargs)
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            return output_text.strip()

        except Exception as e:
            print(f"[MT] Error: {e}")
            return ""


class DummyMT(MTModel):
    """Dummy MT for testing. Returns input unchanged or with prefix."""

    def __init__(self, prefix: str = "[MT] "):
        self.prefix = prefix

    def translate(self, text: str) -> str:
        return f"{self.prefix}{text}"


MarianMT = PipelineMT
M2M100MT = PipelineMT


def create_mt_model(
    model_id: str = "Helsinki-NLP/opus-mt-en-de",
    device: Optional[str] = None,
    src_lang: Optional[str] = None,
    tgt_lang: Optional[str] = None,
    offline: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> MTModel:
    """Factory function to create MT model from config."""
    if model_id == "dummy":
        return DummyMT(**kwargs)

    return PipelineMT(
        model_id=model_id,
        device=device,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        offline=offline,
        cache_dir=cache_dir,
        **kwargs,
    )
