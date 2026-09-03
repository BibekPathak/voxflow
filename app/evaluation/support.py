from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

import numpy as np

from app.audio.resampling import float32_to_pcm16_bytes
from app.config import Settings
from app.providers.factory import ProviderSet
from app.providers.llm.mock import MockLLMProvider
from app.providers.tts.mock import MockTTSProvider
from app.providers.types import AudioData, Transcript
from app.runtime.events import EventType, TranscriptPartial, VoiceEvent
from app.runtime.orchestrator import SessionRuntime
from app.runtime.state_machine import RuntimeState
from app.tools.registry import ToolRegistry

_FRAME = 320


def eval_settings(**overrides: object) -> Settings:
    """Fast, deterministic settings so evaluation scenarios run in near-real-time."""
    base: dict[str, object] = dict(
        sample_rate=16_000,
        vad_speech_start_rms=0.01,
        vad_speech_end_rms=0.005,
        vad_start_confirm_ms=40,
        vad_end_confirm_ms=80,
        vad_frame_ms=20,
        turn_min_speech_ms=100,
        turn_silence_ms=200,
        turn_max_utterance_s=30,
        audio_queue_maxsize=1024,
        provider_stt="mock",
        provider_llm="mock",
        provider_tts="mock",
        mock_llm_first_token_ms=5.0,
        mock_llm_token_interval_ms=1.0,
        mock_tts_first_audio_ms=10.0,
        database_url="sqlite+aiosqlite:///:memory:",
        context_max_turns=10,
        context_max_chars=8000,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def tone(seconds: float, amplitude: float = 0.3, freq: float = 440.0) -> np.ndarray:
    n = int(seconds * 16_000)
    t = np.arange(n, dtype=np.float64) / 16_000
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


def pcm(samples: np.ndarray) -> bytes:
    return float32_to_pcm16_bytes(samples)


def frames(samples: np.ndarray) -> list[bytes]:
    return [pcm(samples[i : i + _FRAME]) for i in range(0, len(samples), _FRAME)]


async def wait_until(condition, timeout: float) -> bool:
    try:
        async with asyncio.timeout(timeout):
            while not condition():
                await asyncio.sleep(0.01)
        return True
    except TimeoutError:
        return False


class NoopSTT:
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


class SlowTTS:
    """Paces audio so the agent is genuinely still speaking when the user
    interrupts, giving the interruption scenarios a real window to act in."""

    def __init__(self, *, frames_count: int = 400, frame_gap_ms: float = 20) -> None:
        self.frames_count = frames_count
        self.frame_gap_ms = frame_gap_ms

    async def synthesize_stream(
        self, text: AsyncIterator[str], *, sample_rate: int = 16_000
    ) -> AsyncIterator[AudioData]:
        async for _ in text:
            pass
        frame = pcm(silence(0.08))
        for _ in range(self.frames_count):
            await asyncio.sleep(self.frame_gap_ms / 1000.0)
            yield AudioData(pcm=frame, sample_rate=sample_rate)

    async def synthesize(self, text: str, *, sample_rate: int = 16_000) -> AudioData:
        return AudioData(pcm=b"", sample_rate=sample_rate)


class RecorderOutbound:
    def __init__(self) -> None:
        self.text: list[dict] = []
        self.audio: list[bytes] = []

    async def send_text(self, data: str) -> None:
        self.text.append(json.loads(data))

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)

    @property
    def audio_frames(self) -> int:
        return len(self.audio)


class SessionHarness:
    """Drives a SessionRuntime in-process for evaluation scenarios.

    Audio is synthesized tone/silence (to trigger VAD) while transcripts are
    injected directly, so scenario content is deterministic and provider-free.
    """

    def __init__(
        self,
        *,
        settings_overrides: dict[str, object] | None = None,
        slow_tts: bool = False,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.settings = eval_settings(**(settings_overrides or {}))
        stt = NoopSTT()
        llm = MockLLMProvider()
        tts = SlowTTS() if slow_tts else MockTTSProvider(first_audio_ms=10.0)
        self.runtime = SessionRuntime(
            session_id=f"eval_{uuid.uuid4().hex[:12]}",
            conversation_id=f"conv_{uuid.uuid4().hex[:12]}",
            settings=self.settings,
            providers=ProviderSet(stt=stt, llm=llm, tts=tts),
            tools=tools,
        )
        self.events: list[VoiceEvent] = []
        self.runtime.bus.subscribe(lambda e: self.events.append(e), session_id=self.runtime.session_id)
        self.outbound = RecorderOutbound()

    async def open(self) -> None:
        self.runtime.attach("eval", outbound=self.outbound)

    async def close(self) -> None:
        await self.runtime.detach("eval")

    def of_type(self, event_type: EventType) -> list[VoiceEvent]:
        return [e for e in self.events if e.type is event_type]

    def latencies(self) -> dict:
        return self.runtime.metrics.snapshot()["latencies_ms"]

    def counters(self) -> dict:
        return self.runtime.metrics.snapshot()["counters"]

    @property
    def state(self) -> RuntimeState:
        return self.runtime.state

    async def feed(self, data: bytes, *, sequence: int | None = None) -> None:
        await self.runtime.ingest_audio(data, sequence=sequence)

    async def burst(self, seconds: float, amplitude: float = 0.3) -> None:
        for chunk in frames(tone(seconds, amplitude=amplitude)):
            await self.feed(chunk)

    async def silence_pause(self, seconds: float) -> None:
        for chunk in frames(silence(seconds)):
            await self.feed(chunk)

    async def say(self, text: str, *, tone_seconds: float = 0.25, silence_seconds: float = 0.5) -> None:
        await self.burst(tone_seconds)
        await self.runtime.bus.publish_and_await(
            TranscriptPartial(
                session_id=self.runtime.session_id,
                text=text,
                timestamp=time.time(),
            )
        )
        await self.silence_pause(silence_seconds)

    async def wait_turns(self, count: int, timeout: float = 5.0) -> bool:
        return await wait_until(lambda: len(self.of_type(EventType.TURN_COMPLETED)) >= count, timeout)

    def last_completed(self) -> VoiceEvent | None:
        completed = self.of_type(EventType.TURN_COMPLETED)
        return completed[-1] if completed else None

    def audio_frame_count_after(self, event: VoiceEvent) -> int:
        return len(self.outbound.audio)
