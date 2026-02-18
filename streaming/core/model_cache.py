"""
Model caching utilities for reproducible model loading.

Provides local caching of HuggingFace models to ensure reproducibility
across experiments and environments.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable

from .config import MODEL_CACHE_DIR

logger = logging.getLogger(__name__)

# Set offline mode for HuggingFace early to avoid HTTP requests
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

# Cache directories
HF_SNAPSHOT_DIR = MODEL_CACHE_DIR / "hf_snapshots"
EVALUATE_CACHE_DIR = MODEL_CACHE_DIR / "evaluate"
COMET_CACHE_DIR = EVALUATE_CACHE_DIR / "comet"
COMET_QE_CACHE_DIR = EVALUATE_CACHE_DIR / "comet_qe"

# In-memory cache for loaded models
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}


def get_hf_cache_dir() -> Path:
    """Get HuggingFace model cache directory."""
    HF_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return HF_SNAPSHOT_DIR


def get_comet_cache_dir(model_name: str = "wmt22-comet-da") -> Path:
    """Get COMET model cache directory."""
    cache_dir = COMET_CACHE_DIR / model_name.replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_comet_qe_cache_dir(model_name: str = "wmt20-comet-qe-da") -> Path:
    """Get COMET-QE model cache directory."""
    cache_dir = COMET_QE_CACHE_DIR / model_name.replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def ensure_model_snapshot(
    model_id: str,
    revision: str = "main",
    force_download: bool = False,
) -> Path:
    """
    Ensure a HuggingFace model is downloaded to local cache.
    """
    from huggingface_hub import snapshot_download

    cache_dir = get_hf_cache_dir()

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=not force_download and _model_exists_locally(model_id, cache_dir),
        )
        logger.info("Model %s available at %s", model_id, local_path)
        return Path(local_path)
    except Exception as e:
        if not force_download:
            logger.warning("Local model not found, downloading %s...", model_id)
            return ensure_model_snapshot(model_id, revision, force_download=True)
        raise RuntimeError(f"Failed to download model {model_id}: {e}")


def _model_exists_locally(model_id: str, cache_dir: Path) -> bool:
    """Check if model exists in local cache."""
    model_folder = cache_dir / f"models--{model_id.replace('/', '--')}"
    return model_folder.exists() and any(model_folder.iterdir())


def get_cached_model(model_id: str) -> Optional[Any]:
    """Get model from in-memory cache."""
    return _MODEL_CACHE.get(model_id)


def set_cached_model(model_id: str, model: Any) -> None:
    """Store model in in-memory cache."""
    _MODEL_CACHE[model_id] = model


def get_cached_tokenizer(model_id: str) -> Optional[Any]:
    """Get tokenizer from in-memory cache."""
    return _TOKENIZER_CACHE.get(model_id)


def set_cached_tokenizer(model_id: str, tokenizer: Any) -> None:
    """Store tokenizer in in-memory cache."""
    _TOKENIZER_CACHE[model_id] = tokenizer


def clear_model_cache() -> None:
    """Clear in-memory model cache."""
    _MODEL_CACHE.clear()
    _TOKENIZER_CACHE.clear()


def load_model_and_tokenizer(
    model_id: str,
    model_class: type,
    tokenizer_class: type,
    device: Optional[str] = None,
    use_cache: bool = True,
    **model_kwargs,
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer with caching.
    """
    import torch

    if use_cache:
        cached_model = get_cached_model(model_id)
        cached_tokenizer = get_cached_tokenizer(model_id)
        if cached_model is not None and cached_tokenizer is not None:
            logger.debug("Using cached model: %s", model_id)
            return cached_model, cached_tokenizer

    cache_dir = get_hf_cache_dir()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading %s on %s", model_id, device)

    tokenizer = tokenizer_class.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    model = model_class.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        **model_kwargs,
    )

    if device != "cpu":
        model = model.to(device)

    model.eval()

    if use_cache:
        set_cached_model(model_id, model)
        set_cached_tokenizer(model_id, tokenizer)

    return model, tokenizer


def safe_call_with_retry(
    fn: Callable,
    *args,
    max_retries: int = 3,
    retry_delay: float = 0.1,
    retry_exceptions: tuple = (RuntimeError, ValueError),
    **kwargs,
) -> Any:
    """
    Call function with retry logic.
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except retry_exceptions as e:
            last_exception = e
            logger.warning("Attempt %s/%s failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))

    raise RuntimeError(f"All {max_retries} attempts failed") from last_exception


def safe_translate_single(
    translate_fn: Callable[[str], str],
    text: str,
    max_retries: int = 3,
    fallback: str = "",
) -> str:
    """
    Safely translate a single text with retry and fallback.
    """
    if not text or not text.strip():
        return fallback

    try:
        return safe_call_with_retry(
            translate_fn,
            text,
            max_retries=max_retries,
            retry_exceptions=(RuntimeError, ValueError, IndexError),
        )
    except Exception as e:
        logger.error("Translation failed after %s retries: %s", max_retries, e)
        return fallback


def batch_translate(
    translate_fn: Callable[[str], str],
    texts: list,
    max_retries: int = 3,
    retry_delay: float = 0.1,
) -> list:
    """
    Translate a list of texts with retry on failures.
    """
    outputs = []
    for text in texts:
        out = safe_translate_single(
            translate_fn,
            text,
            max_retries=max_retries,
            fallback="",
        )
        outputs.append(out)
        if retry_delay > 0:
            time.sleep(retry_delay)
    return outputs
