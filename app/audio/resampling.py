from __future__ import annotations

import numpy as np

INT16_MAX = 32768.0


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Decode little-endian 16-bit PCM bytes into float32 samples in [-1, 1]."""
    raw = np.frombuffer(data, dtype="<i2")
    return raw.astype(np.float32) / INT16_MAX


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Encode float32 samples in [-1, 1] into little-endian 16-bit PCM bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * (INT16_MAX - 1)).astype("<i2")
    return pcm.tobytes()


def to_mono_float32(samples: np.ndarray, channels: int = 1) -> np.ndarray:
    """Convert a (N, channels) or flat interleaved buffer to mono float32.

    For 2D input each channel is averaged. Flat input with channels > 1 is
    assumed interleaved and averaged across channels.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        if channels == 1:
            return samples
        if samples.size % channels != 0:
            raise ValueError("flat buffer length is not divisible by channel count")
        samples = samples.reshape(-1, channels)
    if samples.ndim != 2:
        raise ValueError("expected 1D or 2D audio buffer")
    return samples.mean(axis=1).astype(np.float32)


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1D float32 buffer by linear interpolation between samples."""
    if src_rate == dst_rate:
        return samples
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    n_out = max(1, round(samples.size * dst_rate / src_rate))
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    src_positions = (
        np.arange(n_out, dtype=np.float64) * (samples.size - 1) / (n_out - 1) if n_out > 1 else np.array([0.0])
    )
    return np.interp(src_positions, np.arange(samples.size, dtype=np.float64), samples).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    """Root-mean-square amplitude of a float32 buffer (0..1 range)."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.asarray(samples, dtype=np.float64)))))
