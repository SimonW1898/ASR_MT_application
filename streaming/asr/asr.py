"""
ASR model wrappers for the streaming pipeline.

Uses HuggingFace models with local caching for offline use.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Optional, Any, Tuple
import numpy as np

import torch

from ..core.types import ASROutput, DEFAULT_SR
from ..models import load_asr_model_and_processor, HF_SNAPSHOT_ROOT


class ASRModel(ABC):
    """Abstract interface for ASR models."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int = DEFAULT_SR) -> ASROutput:
        """Transcribe audio to text."""
        raise NotImplementedError()


@lru_cache(maxsize=4)
def _get_asr_model_and_processor(
    model_id: str,
    offline: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Get or create cached ASR model and processor from local snapshot."""
    cache_path = Path(cache_dir) if cache_dir else HF_SNAPSHOT_ROOT

    print(f"[ASR] Loading model for {model_id} (offline={offline})")
    return load_asr_model_and_processor(
        model_id,
        offline=offline,
        cache_dir=cache_path,
    )


class PipelineASR(ASRModel):
    """Generic ASR using HuggingFace models with local caching."""

    def __init__(
        self,
        model_id: str = "openai/whisper-small",
        device: Optional[str] = None,
        language: Optional[str] = "en",
        task: str = "transcribe",
        chunk_length_s: Optional[float] = None,
        offline: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.language = language
        self.task = task
        self.chunk_length_s = chunk_length_s

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model, self.processor = _get_asr_model_and_processor(
            model_id, offline, cache_dir
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[ASR] Model ready: {model_id}")

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sr: int = DEFAULT_SR) -> ASROutput:
        if audio.ndim != 1:
            raise ValueError("audio must be mono (1D array)")

        if len(audio) == 0:
            return ASROutput(text="", extra={"empty_input": True})

        try:
            inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            generate_kwargs = {"max_new_tokens": 256}
            if "whisper" in self.model_id.lower():
                if self.language:
                    generate_kwargs["language"] = self.language
                generate_kwargs["task"] = self.task

            predicted_ids = self.model.generate(**inputs, **generate_kwargs)
            text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            return ASROutput(text=text.strip(), extra={})

        except Exception as e:
            print(f"[ASR] Error: {e}")
            return ASROutput(text="", extra={"error": str(e)})


class DummyASR(ASRModel):
    """Dummy ASR for testing. Returns a fixed string."""

    def __init__(self, fixed_text: str = "Hello world"):
        self.fixed_text = fixed_text

    def transcribe(self, audio: np.ndarray, sr: int = DEFAULT_SR) -> ASROutput:
        return ASROutput(text=self.fixed_text, extra={"dummy": True})


WhisperASR = PipelineASR
Wav2VecASR = PipelineASR


def create_asr_model(
    model_id: str = "openai/whisper-small",
    device: Optional[str] = None,
    offline: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> ASRModel:
    """Factory function to create ASR model from config."""
    if model_id == "dummy":
        return DummyASR(**kwargs)

    return PipelineASR(
        model_id=model_id,
        device=device,
        offline=offline,
        cache_dir=cache_dir,
        **kwargs,
    )
