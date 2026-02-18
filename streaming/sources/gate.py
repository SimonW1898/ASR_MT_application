"""
Speech gate: Voice Activity Detection (VAD) abstractions.
"""

from abc import ABC, abstractmethod
import numpy as np


class SpeechGate(ABC):
    """Abstract interface for speech/silence detection."""

    @abstractmethod
    def is_speech(self, frame_audio: np.ndarray) -> bool:
        """Determine if the audio frame contains speech."""
        raise NotImplementedError()


class RMSGate(SpeechGate):
    """Simple RMS-based voice activity detector."""

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def is_speech(self, frame_audio: np.ndarray) -> bool:
        rms = self._rms(frame_audio)
        return rms >= self.threshold

    def _rms(self, audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))


class EnergyGate(SpeechGate):
    """Energy-based gate with adaptive threshold."""

    def __init__(
        self,
        min_threshold: float = 0.005,
        ratio_threshold: float = 3.0,
        alpha: float = 0.1,
    ):
        self.min_threshold = min_threshold
        self.ratio_threshold = ratio_threshold
        self.alpha = alpha
        self.noise_floor = min_threshold

    def is_speech(self, frame_audio: np.ndarray) -> bool:
        energy = self._energy(frame_audio)
        threshold = max(self.min_threshold, self.noise_floor * self.ratio_threshold)
        is_speech = energy >= threshold

        if not is_speech:
            self.noise_floor = (1 - self.alpha) * self.noise_floor + self.alpha * energy

        return is_speech

    def _energy(self, audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def reset(self) -> None:
        self.noise_floor = self.min_threshold
