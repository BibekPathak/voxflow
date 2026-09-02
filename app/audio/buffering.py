from __future__ import annotations

import numpy as np


class AudioBuffer:
    """Contiguous in-memory accumulation of mono float32 samples.

    Used to accumulate a recorded speech segment before it is handed to a
    non-streaming consumer (e.g. an STT provider that needs a full buffer).
    """

    def __init__(self, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self._samples: np.ndarray = np.zeros(0, dtype=np.float32)

    def append(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return
        if samples.ndim != 1:
            raise ValueError("AudioBuffer expects mono float32 samples")
        self._samples = np.concatenate([self._samples, samples])

    def clear(self) -> None:
        self._samples = np.zeros(0, dtype=np.float32)

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    @property
    def num_samples(self) -> int:
        return int(self._samples.size)

    @property
    def duration_ms(self) -> float:
        return self.num_samples / self.sample_rate * 1000.0

    def drain(self) -> np.ndarray:
        out = self._samples
        self._samples = np.zeros(0, dtype=np.float32)
        return out

    def __len__(self) -> int:
        return self.num_samples


class SequenceGapDetector:
    """Detects dropped or reordered audio frames by sequence number.

    Streams of network-delivered audio chunks should arrive in order but may
    drop out under degraded conditions. Each audio frame carries a monotonically
    increasing sequence; this detector reports discontinuities so the runtime can
    account for lost audio instead of misinterpreting a gap as an endpoint.
    """

    def __init__(self, *, warn_on_gap: bool = True) -> None:
        self._next_expected: int | None = None
        self.warn_on_gap = warn_on_gap
        self.total_gaps = 0
        self.gaps: list[tuple[int, int]] = []

    def observe(self, sequence: int) -> None:
        if self._next_expected is None:
            self._next_expected = sequence + 1
            return
        if sequence < self._next_expected:
            return
        if sequence > self._next_expected:
            self.total_gaps += 1
            self.gaps.append((self._next_expected, sequence - 1))
        self._next_expected = sequence + 1

    @property
    def missing_frames(self) -> int:
        return sum(hi - lo + 1 for lo, hi in self.gaps)

    def reset(self) -> None:
        self._next_expected = None
        self.total_gaps = 0
        self.gaps = []
