from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from app.providers.types import AudioData, LLMChunk, LLMMessage, ToolSpec, Transcript


class STTProvider(Protocol):
    """Streaming speech-to-text.

    ``transcribe_stream`` consumes an async stream of raw little-endian 16-bit
    PCM chunks (mono, ``sample_rate``) and yields transcripts as they become
    available: interim partial results first, then a final transcript once the
    provider decides the utterance is complete.
    """

    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        sample_rate: int = 16_000,
        interim_results: bool = True,
        language: str | None = None,
    ) -> AsyncIterator[Transcript]: ...


class LLMProvider(Protocol):
    """Streaming large-language-model chat.

    ``stream_chat`` streams deltas: plain text tokens and/or function/tool-call
    fragments (each with an index so partial tool-call arguments can be
    reassembled across chunks). A terminal chunk carries the finish reason.
    """

    async def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[LLMChunk]: ...


class TTSProvider(Protocol):
    """Streaming text-to-speech.

    ``synthesize_stream`` consumes an async stream of text fragments and yields
    raw 16-bit PCM audio (mono) as it is produced, so the first audio can be
    played before the full sentence has been synthesized.
    """

    async def synthesize_stream(
        self,
        text: AsyncIterator[str],
        *,
        sample_rate: int = 16_000,
    ) -> AsyncIterator[AudioData]: ...

    async def synthesize(self, text: str, *, sample_rate: int = 16_000) -> AudioData: ...


class AsyncCloseable(Protocol):
    async def close(self) -> None: ...
