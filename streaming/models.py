"""
Unified model caching and loading for the streaming pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from huggingface_hub import snapshot_download

from .core.model_cache import (
    HF_SNAPSHOT_DIR,
    EVALUATE_CACHE_DIR,
    get_cached_model,
    set_cached_model,
    get_cached_tokenizer,
    set_cached_tokenizer,
)

# Module-level defaults
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_SNAPSHOT_ROOT = HF_SNAPSHOT_DIR
EVAL_CACHE_DIR = EVALUATE_CACHE_DIR

_HF_LOADED_ASR: Dict[str, Tuple[Any, Any]] = {}
_HF_LOADED_MT: Dict[str, Tuple[Any, Any]] = {}


def _safe_dirname(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace("\\", "_")


def ensure_hf_snapshot(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    offline: bool = False,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Ensure the full HF repo snapshot is present in our project cache."""
    cache = cache_dir or HF_SNAPSHOT_ROOT
    cache.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache),
        local_files_only=offline,
    )
    return Path(local_dir)


def load_asr_model_and_processor(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    offline: bool = False,
    cache_dir: Optional[Path] = None,
) -> Tuple[Any, Any]:
    """Load ASR model + processor from local snapshot directory."""
    key = f"{repo_id}@{revision or 'default'}"
    if key in _HF_LOADED_ASR:
        return _HF_LOADED_ASR[key]

    local_dir = ensure_hf_snapshot(
        repo_id,
        revision=revision,
        offline=offline,
        cache_dir=cache_dir,
    )

    if "whisper" in repo_id.lower():
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        processor = WhisperProcessor.from_pretrained(str(local_dir), local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(str(local_dir), local_files_only=True)
    else:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        processor = Wav2Vec2Processor.from_pretrained(str(local_dir), local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(str(local_dir), local_files_only=True)

    _HF_LOADED_ASR[key] = (model, processor)
    return model, processor


def load_mt_model_and_tokenizer(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    offline: bool = False,
    cache_dir: Optional[Path] = None,
) -> Tuple[Any, Any]:
    """Load MT model + tokenizer from local snapshot directory."""
    key = f"{repo_id}@{revision or 'default'}"
    if key in _HF_LOADED_MT:
        return _HF_LOADED_MT[key]

    local_dir = ensure_hf_snapshot(
        repo_id,
        revision=revision,
        offline=offline,
        cache_dir=cache_dir,
    )

    if "m2m100" in repo_id.lower():
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        tokenizer = M2M100Tokenizer.from_pretrained(str(local_dir), local_files_only=True)
        model = M2M100ForConditionalGeneration.from_pretrained(str(local_dir), local_files_only=True)
    elif "nllb" in repo_id.lower():
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(local_dir), local_files_only=True)
    else:
        from transformers import MarianMTModel, MarianTokenizer
        tokenizer = MarianTokenizer.from_pretrained(str(local_dir), local_files_only=True)
        model = MarianMTModel.from_pretrained(str(local_dir), local_files_only=True)

    _HF_LOADED_MT[key] = (model, tokenizer)
    return model, tokenizer


def build_asr_pipeline(
    repo_id: str,
    *,
    device: Optional[str] = None,
    offline: bool = False,
    cache_dir: Optional[Path] = None,
    dtype: Optional[torch.dtype] = None,
    **pipe_kwargs,
) -> Any:
    """Build ASR pipeline from cached model."""
    from transformers import pipeline

    use_gpu = (device == "cuda") or (device is None and torch.cuda.is_available())
    device_index = 0 if use_gpu else -1

    if dtype is None and use_gpu:
        dtype = torch.float16

    print(f"[ASR] Loading {repo_id} from cache...")
    model, processor = load_asr_model_and_processor(
        repo_id, offline=offline, cache_dir=cache_dir
    )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,
        feature_extractor=processor.feature_extractor if hasattr(processor, "feature_extractor") else processor,
        device=device_index,
        torch_dtype=dtype,
        **pipe_kwargs,
    )
    return pipe


def build_mt_pipeline(
    repo_id: str,
    *,
    device: Optional[str] = None,
    offline: bool = False,
    cache_dir: Optional[Path] = None,
    dtype: Optional[torch.dtype] = None,
    **pipe_kwargs,
) -> Any:
    """Build MT pipeline from cached model."""
    from transformers import pipeline

    use_gpu = (device == "cuda") or (device is None and torch.cuda.is_available())
    device_index = 0 if use_gpu else -1

    if dtype is None and use_gpu:
        dtype = torch.float16

    print(f"[MT] Loading {repo_id} from cache...")
    model, tokenizer = load_mt_model_and_tokenizer(
        repo_id, offline=offline, cache_dir=cache_dir
    )

    pipe = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        device=device_index,
        torch_dtype=dtype,
        **pipe_kwargs,
    )
    return pipe
