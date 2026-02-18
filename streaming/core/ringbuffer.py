"""
Ring buffer utility for efficient audio buffering.

Provides O(1) append and O(n) slice operations for sliding window buffers.
"""

import numpy as np


class RingBuffer:
    """
    Fixed-capacity ring buffer for float32 audio samples.
    """

    def __init__(self, capacity_samples: int):
        """
        Initialize ring buffer.

        Args:
            capacity_samples: Maximum number of samples to store.
        """
        self.capacity = capacity_samples
        self.data = np.zeros(capacity_samples, dtype=np.float32)
        self.write_pos = 0
        self.size = 0

    def append(self, x: np.ndarray) -> None:
        """
        Append samples to the buffer.

        If the new data exceeds capacity, oldest samples are overwritten.
        """
        x = np.asarray(x, dtype=np.float32)
        n = len(x)

        if n >= self.capacity:
            self.data[:] = x[-self.capacity:]
            self.write_pos = 0
            self.size = self.capacity
            return

        end_pos = self.write_pos + n
        if end_pos <= self.capacity:
            self.data[self.write_pos:end_pos] = x
        else:
            first_part = self.capacity - self.write_pos
            self.data[self.write_pos:] = x[:first_part]
            self.data[:n - first_part] = x[first_part:]

        self.write_pos = end_pos % self.capacity
        self.size = min(self.size + n, self.capacity)

    def slice_last(self, n_samples: int) -> np.ndarray:
        """Extract the last n_samples from the buffer."""
        n = min(n_samples, self.size)
        if n == 0:
            return np.array([], dtype=np.float32)

        start_pos = (self.write_pos - n) % self.capacity

        if start_pos + n <= self.capacity:
            return self.data[start_pos:start_pos + n].copy()

        first_part = self.capacity - start_pos
        return np.concatenate([
            self.data[start_pos:],
            self.data[:n - first_part]
        ])

    def get_all(self) -> np.ndarray:
        """Get all samples currently in the buffer."""
        return self.slice_last(self.size)

    def clear(self) -> None:
        """Clear the buffer."""
        self.write_pos = 0
        self.size = 0

    def get_size(self) -> int:
        """Return current number of samples in buffer."""
        return self.size

    def __len__(self) -> int:
        return self.size
