from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import numpy as np

from app.audio.resampling import float32_to_pcm16_bytes
from app.providers.factory import ProviderSet
from app.providers.llm.mock import MockLLMProvider
from app.providers.types import AudioData
from app.runtime.events import EventType, TranscriptPartial, VoiceEvent
from app.runtime.orchestrator import SessionRuntime
from app.runtime.state_machine import RuntimeState
from tests.conftest import make_noop_providers, make_settings
from tests.unit.test_vad import silence_samples, tone_samples


class RecorderOutbound:
    def __init__(self) -> None:
        self.text: list[dict] = []
        self.audio: list[bytes] = []

    async def send_text(self, data: str) -> None:
        self.text.append(json.loads(data))

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)

    @property
    def start_messages(self) -> list[dict]:
        return [m for m in self.text if m["type"] == "agent_audio.start"]

    @property
    def end_messages(self) -> list[dict]:
        return [m for m in self.text if m["type"] == "agent_audio.end"]


class SlowTTSProvider:
    """Yields a long stream of audio frames after consuming the text stream, so
    there is a wide window to interrupt the agent mid-speech."""

    def __init__(self, *, frame_ms: float = 80, frame_gap_ms: float = 20, frames: int = 400) -> None:
        self.frame_samples = int(16_000 * frame_ms / 1000)
        self.frame_gap_ms = frame_gap_ms
        self.frames = frames

    async def synthesize_stream(
        self, text: AsyncIterator[str], *, sample_rate: int = 16_000
    ) -> AsyncIterator[AudioData]:
        async for _ in text:
            pass
        silence = float32_to_pcm16_bytes(np.zeros(self.frame_samples, dtype=np.float32))
        for _ in range(self.frames):
            await asyncio.sleep(self.frame_gap_ms / 1000.0)
            yield AudioData(pcm=silence, sample_rate=sample_rate)

    async def synthesize(self, text: str, *, sample_rate: int = 16_000) -> AudioData:
        return AudioData(pcm=b"", sample_rate=sample_rate)


def _pcm(samples) -> bytes:
    return float32_to_pcm16_bytes(samples)


def _frames(samples) -> list[bytes]:
    frame = 320
    return [_pcm(samples[i : i + frame]) for i in range(0, samples.size, frame)]


async def _wait_until(condition, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


def _events_of(collector: list[VoiceEvent], event_type: EventType) -> list[VoiceEvent]:
    return [e for e in collector if e.type is event_type]


async def _feed(runtime: SessionRuntime, chunks: list[bytes]) -> None:
    for chunk in chunks:
        await runtime.ingest_audio(chunk)


async def test_streaming_turn_end_to_end_with_stt() -> None:
    runtime = SessionRuntime("s1", "c1", make_settings())
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")
    outbound = RecorderOutbound()
    assert runtime.attach("owner-a", outbound=outbound) is True

    await _feed(runtime, _frames(tone_samples(0.5, amplitude=0.3)))
    await _feed(runtime, _frames(silence_samples(0.6)))

    await _wait_until(
        lambda: (
            bool(_events_of(collector, EventType.TURN_COMPLETED))
            and any(m.get("type") == "agent_audio.end" for m in outbound.text)
        )
    )

    started = _events_of(collector, EventType.TURN_STARTED)
    assert started and started[0].turn_id == 1
    assert any(isinstance(e, TranscriptPartial) for e in _events_of(collector, EventType.TRANSCRIPT_PARTIAL))
    assert _events_of(collector, EventType.TRANSCRIPT_FINAL)
    assert _events_of(collector, EventType.LLM_STARTED)
    assert _events_of(collector, EventType.LLM_TOKEN)
    assert _events_of(collector, EventType.TTS_STARTED)
    assert _events_of(collector, EventType.TTS_AUDIO)
    tts_completed = _events_of(collector, EventType.TTS_COMPLETED)
    assert tts_completed and tts_completed[-1].reason == "completed"

    completed = _events_of(collector, EventType.TURN_COMPLETED)[-1]
    assert completed.outcome == "completed"

    assert outbound.audio
    assert outbound.start_messages
    assert outbound.end_messages[-1]["reason"] == "completed"

    states = [e.to_state for e in _events_of(collector, EventType.RUNTIME_STATE_CHANGED)]
    assert "speaking" in states
    assert runtime.state is RuntimeState.LISTENING
    assert runtime.turn_count == 1
    await runtime.detach("owner-a")


async def test_stt_final_precedes_turn_start() -> None:
    runtime = SessionRuntime("s1", "c1", make_settings())
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")
    runtime.attach("owner-a")

    await _feed(runtime, _frames(tone_samples(0.5, amplitude=0.3)))
    await _feed(runtime, _frames(silence_samples(0.6)))
    await _wait_until(lambda: bool(_events_of(collector, EventType.TURN_COMPLETED)))

    finals = _events_of(collector, EventType.TRANSCRIPT_FINAL)
    turn_starts = _events_of(collector, EventType.TURN_STARTED)
    assert finals and turn_starts
    assert collector.index(turn_starts[0]) > collector.index(finals[-1])
    await runtime.detach("owner-a")


async def test_barge_in_stops_agent_audio_without_stale_frames() -> None:
    providers = ProviderSet(stt=make_noop_providers().stt, llm=MockLLMProvider(), tts=SlowTTSProvider())
    runtime = SessionRuntime("s1", "c1", make_settings(), providers=providers)
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")
    outbound = RecorderOutbound()
    runtime.attach("owner-a", outbound=outbound)

    await _feed(runtime, _frames(tone_samples(0.2, amplitude=0.3)))
    await runtime.bus.publish_and_await(TranscriptPartial(session_id="s1", text="please check my account"))
    await _feed(runtime, _frames(silence_samples(0.4)))

    await _wait_until(lambda: bool(outbound.start_messages))
    assert runtime.state is RuntimeState.SPEAKING or runtime.state is RuntimeState.PROCESSING

    await _feed(runtime, _frames(tone_samples(0.2, amplitude=0.3)))
    assert runtime.state is RuntimeState.INTERRUPTED

    await _wait_until(lambda: bool(_events_of(collector, EventType.USER_INTERRUPTED)))
    await _wait_until(
        lambda: (
            bool(_events_of(collector, EventType.TURN_COMPLETED))
            and _events_of(collector, EventType.TURN_COMPLETED)[-1].outcome == "cancelled"
        )
    )
    assert outbound.end_messages and outbound.end_messages[-1]["reason"] == "cancelled"

    frames_after_cancel = len(outbound.audio)
    await asyncio.sleep(0.2)
    assert len(outbound.audio) == frames_after_cancel

    tts_completed = _events_of(collector, EventType.TTS_COMPLETED)
    assert tts_completed and tts_completed[-1].reason == "cancelled"

    await _feed(runtime, _frames(silence_samples(0.3)))
    assert runtime.state is RuntimeState.LISTENING
    await runtime.detach("owner-a")
