from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from app.audio.resampling import float32_to_pcm16_bytes
from app.providers.meta import ProviderInfo
from app.providers.types import AudioData

_MILLIS_PER_CHAR = 60.0
_BASE_MS = 200.0
_FRAME_MS = 80.0


class MockTTSProvider:
    """Offline, deterministic streaming TTS.

    Produces silent PCM whose total duration is a linear function of the input
    text length (roughly human speaking pace). Text fragments are buffered and
    converted to fixed-length audio frames as they arrive, so the provider
    streams audio before the full text is known -- exactly the behavior the
    runtime relies on for low time-to-first-audio.

    ``first_audio_ms`` simulates the network/model latency before the first
    audio frame so latency instrumentation is meaningful under mocks.
    """

    def __init__(self, *, first_audio_ms: float = 0.0, sample_rate: int = 16_000) -> None:
        self.first_audio_ms = first_audio_ms
        self.sample_rate = sample_rate

    async def close(self) -> None:
        return None

    def metadata(self) -> ProviderInfo:
        return ProviderInfo(
            name="mock", kind="tts", vendor="VoxFlow Mock", model="duration-proportional", streaming=True
        )

    def _estimate_ms(self, text: str) -> float:
        if not text:
            return 0.0
        return _BASE_MS + len(text) * _MILLIS_PER_CHAR

    def _pcm_silence(self, num_samples: int, rate: int) -> bytes:
        return float32_to_pcm16_bytes(np.zeros(num_samples, dtype=np.float32)) if num_samples > 0 else b""

    async def synthesize(self, text: str, *, sample_rate: int = 16_000) -> AudioData:
        rate = sample_rate or self.sample_rate
        num_samples = int(self._estimate_ms(text) / 1000.0 * rate)
        return AudioData(pcm=self._pcm_silence(num_samples, rate), sample_rate=rate)

    async def synthesize_stream(
        self,
        text: AsyncIterator[str],
        *,
        sample_rate: int = 16_000,
    ) -> AsyncIterator[AudioData]:
        rate = sample_rate or self.sample_rate
        frame_samples = int(rate * _FRAME_MS / 1000.0)
        pending_ms = 0.0
        first = True
        started = False

        async for piece in text:
            if not piece:
                continue
            pending_ms += len(piece) * _MILLIS_PER_CHAR
            while pending_ms >= _FRAME_MS:
                pending_ms -= _FRAME_MS
                if first:
                    await asyncio.sleep(self.first_audio_ms / 1000.0)
                    first = False
                started = True
                yield AudioData(pcm=self._pcm_silence(frame_samples, rate), sample_rate=rate)

        if not started:
            return
        if pending_ms > 0:
            num_samples = int(pending_ms / 1000.0 * rate)
            if num_samples > 0:
                yield AudioData(pcm=self._pcm_silence(num_samples, rate), sample_rate=rate)
