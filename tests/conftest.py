from __future__ import annotations

from collections.abc import AsyncIterator

from app.config import Settings
from app.providers.factory import ProviderSet
from app.providers.types import AudioData, LLMChunk, Transcript


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = dict(
        sample_rate=16_000,
        vad_speech_start_rms=0.01,
        vad_speech_end_rms=0.005,
        vad_start_confirm_ms=40,
        vad_end_confirm_ms=80,
        vad_frame_ms=20,
        turn_min_speech_ms=80,
        turn_silence_ms=120,
        turn_max_utterance_s=5,
        audio_queue_maxsize=256,
        provider_stt="mock",
        provider_llm="mock",
        provider_tts="mock",
        database_url="sqlite+aiosqlite:///:memory:",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class NoopSTTProvider:
    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        sample_rate: int = 16_000,
        interim_results: bool = True,
        language: str | None = None,
    ) -> AsyncIterator[Transcript]:
        del sample_rate, interim_results, language
        async for _ in audio:
            pass
        if False:  # pragma: no cover
            yield Transcript(text="")


class NoopLLMProvider:
    async def stream_chat(self, messages, *, tools=None, temperature=None) -> AsyncIterator[LLMChunk]:
        del messages, tools, temperature
        yield LLMChunk(finish_reason="stop")


class NoopTTSProvider:
    async def synthesize_stream(self, text, *, sample_rate: int = 16_000) -> AsyncIterator[AudioData]:
        del text, sample_rate
        if False:  # pragma: no cover
            yield AudioData()

    async def synthesize(self, text: str, *, sample_rate: int = 16_000) -> AudioData:
        del text
        return AudioData(sample_rate=sample_rate)


def make_noop_providers() -> ProviderSet:
    return ProviderSet(stt=NoopSTTProvider(), llm=NoopLLMProvider(), tts=NoopTTSProvider())
