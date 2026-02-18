"""
Audio preprocessing for streaming ASR pipelines.

Provides lightweight band-pass filtering and ML-based denoising.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np

_scipy_signal = None


def _get_scipy_signal():
    """Lazy import scipy.signal."""
    global _scipy_signal
    if _scipy_signal is None:
        from scipy import signal
        _scipy_signal = signal
    return _scipy_signal


class PreprocessMode(str, Enum):
    """Available preprocessing modes."""
    NONE = "none"
    BANDPASS = "bandpass"
    ML = "ml"


@dataclass
class PreprocessConfig:
    """Configuration for audio preprocessing."""
    mode: str = "bandpass"
    ml_model: str = "noisereduce"
    ml_device: str = "cpu"

    highpass_cutoff_hz: int = 80
    lowpass_cutoff_hz: int = 8000
    filter_order: int = 2

    target_rms: float = 0.08

    ml_chunk_s: float = 3.0
    ml_overlap_s: float = 0.2


@dataclass
class PreprocessStats:
    """Statistics from preprocessing."""
    mode: str
    processing_time_ms: float
    rms_before: float = 0.0
    rms_after: float = 0.0
    ml_model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "processing_time_ms": self.processing_time_ms,
            "rms_before": self.rms_before,
            "rms_after": self.rms_after,
            "ml_model_used": self.ml_model_used,
        }


def highpass_filter(
    audio: np.ndarray,
    sr: int = 16000,
    cutoff_hz: int = 80,
    order: int = 2,
) -> np.ndarray:
    """Apply high-pass Butterworth filter to remove low-frequency rumble."""
    if len(audio) < 50:
        return audio

    signal = _get_scipy_signal()

    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist

    b, a = signal.butter(order, normalized_cutoff, btype="high")

    padlen = min(3 * max(len(a), len(b)), len(audio) - 1)
    if padlen < 1:
        return audio

    filtered = signal.filtfilt(b, a, audio, padlen=padlen)

    return filtered.astype(np.float32)


def lowpass_filter(
    audio: np.ndarray,
    sr: int = 16000,
    cutoff_hz: int = 8000,
    order: int = 2,
) -> np.ndarray:
    """Apply low-pass Butterworth filter to remove high-frequency hiss."""
    if len(audio) < 50:
        return audio

    signal = _get_scipy_signal()

    nyquist = sr / 2
    if cutoff_hz >= nyquist:
        cutoff_hz = nyquist * 0.999
    if cutoff_hz <= 0:
        return audio
    normalized_cutoff = cutoff_hz / nyquist

    b, a = signal.butter(order, normalized_cutoff, btype="low")

    padlen = min(3 * max(len(a), len(b)), len(audio) - 1)
    if padlen < 1:
        return audio

    filtered = signal.filtfilt(b, a, audio, padlen=padlen)

    return filtered.astype(np.float32)


def normalize_rms(
    audio: np.ndarray,
    target_rms: float = 0.08,
    max_gain_db: float = 20.0,
) -> np.ndarray:
    """Normalize audio to target RMS level."""
    if target_rms <= 0:
        return audio

    current_rms = np.sqrt(np.mean(audio ** 2))

    if current_rms < 1e-8:
        return audio

    gain = target_rms / current_rms

    max_gain = 10 ** (max_gain_db / 20)
    gain = min(gain, max_gain)

    audio_normalized = audio * gain
    audio_normalized = np.clip(audio_normalized, -1.0, 1.0)

    return audio_normalized.astype(np.float32)


_ML_DENOISERS: Dict[str, Callable] = {}


def register_ml_denoiser(name: str):
    """Decorator to register an ML denoiser function."""
    def decorator(func: Callable):
        _ML_DENOISERS[name] = func
        return func
    return decorator


def get_available_ml_denoisers() -> list:
    """Return list of available ML denoiser names."""
    available = []
    for name in _ML_DENOISERS:
        try:
            _ML_DENOISERS[name](np.zeros(1600, dtype=np.float32), 16000)
            available.append(name)
        except ImportError:
            pass
    return available


@register_ml_denoiser("noisereduce")
def _denoise_noisereduce(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Denoise using noisereduce library."""
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=sr, **kwargs).astype(np.float32)


@register_ml_denoiser("demucs")
def _denoise_demucs(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    """Denoise using demucs."""
    try:
        import torch
        import demucs.pretrained
        import demucs.apply
        import demucs.audio
    except Exception as e:
        raise ImportError("demucs not installed") from e

    model = demucs.pretrained.dns64().eval()

    x = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    out = demucs.apply.apply_model(model, x, split=True, overlap=0.25)
    out = out.squeeze(0).mean(0)

    return out.numpy().astype(np.float32)


@register_ml_denoiser("none")
def _denoise_none(audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
    return audio


def _denoise_ml(
    audio: np.ndarray,
    sr: int,
    config: PreprocessConfig,
) -> np.ndarray:
    if config.ml_model not in _ML_DENOISERS:
        raise ValueError(f"Unknown ML denoiser: {config.ml_model}")

    denoise_fn = _ML_DENOISERS[config.ml_model]

    chunk_samples = int(config.ml_chunk_s * sr)
    overlap_samples = int(config.ml_overlap_s * sr)

    if len(audio) <= chunk_samples:
        return denoise_fn(audio, sr)

    outputs = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        out = denoise_fn(chunk, sr)
        outputs.append(out)
        if end >= len(audio):
            break
        start = end - overlap_samples

    return np.concatenate(outputs).astype(np.float32)


def clean_audio(
    audio: np.ndarray,
    *,
    sr: int = 16000,
    mode: str = "bandpass",
    config: Optional[PreprocessConfig] = None,
) -> Tuple[np.ndarray, PreprocessStats]:
    """
    Clean audio with optional preprocessing.

    Returns cleaned audio and preprocessing stats.
    """
    config = config or PreprocessConfig(mode=mode)
    mode = config.mode

    start = time.time()

    rms_before = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0

    if mode == PreprocessMode.NONE:
        cleaned = audio
    elif mode == PreprocessMode.BANDPASS:
        cleaned = highpass_filter(audio, sr=sr, cutoff_hz=config.highpass_cutoff_hz, order=config.filter_order)
        cleaned = lowpass_filter(cleaned, sr=sr, cutoff_hz=config.lowpass_cutoff_hz, order=config.filter_order)
        cleaned = normalize_rms(cleaned, target_rms=config.target_rms)
    elif mode == PreprocessMode.ML:
        cleaned = _denoise_ml(audio, sr=sr, config=config)
        cleaned = normalize_rms(cleaned, target_rms=config.target_rms)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    rms_after = float(np.sqrt(np.mean(cleaned ** 2))) if len(cleaned) > 0 else 0.0

    stats = PreprocessStats(
        mode=mode,
        processing_time_ms=(time.time() - start) * 1000.0,
        rms_before=rms_before,
        rms_after=rms_after,
        ml_model_used=config.ml_model if mode == PreprocessMode.ML else None,
    )

    return cleaned.astype(np.float32), stats
