from __future__ import annotations

import asyncio

from app.audio.resampling import float32_to_pcm16_bytes
from app.runtime.events import (
    EventType,
    TranscriptPartial,
    VoiceEvent,
)
from app.runtime.orchestrator import SessionRuntime, TurnContext
from app.runtime.state_machine import RuntimeState
from tests.conftest import make_noop_providers, make_settings
from tests.unit.test_vad import silence_samples, tone_samples


def _pcm(samples) -> bytes:
    return float32_to_pcm16_bytes(samples)


async def _wait_until(condition, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


def _events_of(collector: list[VoiceEvent], event_type: EventType) -> list[VoiceEvent]:
    return [e for e in collector if e.type is event_type]


async def test_attach_detach_lifecycle() -> None:
    runtime = SessionRuntime("s1", "c1", make_settings())
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")

    assert runtime.state is RuntimeState.IDLE
    assert runtime.attach("owner-a") is True
    assert runtime.state is RuntimeState.LISTENING
    assert runtime.audio_connected

    assert runtime.attach("owner-b") is False

    await runtime.detach("owner-a")
    assert runtime.state is RuntimeState.CLOSED
    assert runtime.audio_connected is False
    assert runtime.attach("owner-c") is False


async def test_barge_in_interrupts_speaking_and_returns_to_listening() -> None:
    settings = make_settings()
    runtime = SessionRuntime("s1", "c1", settings, providers=make_noop_providers())
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")

    runtime.attach("owner-a")
    runtime.states.transition(RuntimeState.PROCESSING, reason="test")
    runtime.states.transition(RuntimeState.SPEAKING, reason="test")

    await runtime.ingest_audio(_pcm(tone_samples(0.2, amplitude=0.3)))
    assert runtime.state is RuntimeState.INTERRUPTED
    await _wait_until(lambda: bool(_events_of(collector, EventType.USER_INTERRUPTED)))

    await runtime.ingest_audio(_pcm(silence_samples(0.3)))
    assert runtime.state is RuntimeState.LISTENING
    await runtime.detach("owner-a")


async def test_full_user_turn_through_detector_pipeline() -> None:
    runtime = SessionRuntime("s1", "c1", make_settings())
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")
    runtime.attach("owner-a")

    await runtime.ingest_audio(_pcm(tone_samples(0.2, amplitude=0.3)))
    await runtime.bus.publish_and_await(TranscriptPartial(session_id="s1", turn_id=None, text="hello there."))
    await runtime.ingest_audio(_pcm(silence_samples(0.3)))

    await _wait_until(lambda: bool(_events_of(collector, EventType.TURN_COMPLETED)))

    assert _events_of(collector, EventType.TURN_STARTED)
    completed = _events_of(collector, EventType.TURN_COMPLETED)[-1]
    assert completed.outcome == "completed"
    assert runtime.turn_count == 1
    assert runtime.current_turn_id is None
    assert runtime.state is RuntimeState.LISTENING
    await runtime.detach("owner-a")


async def test_barge_in_cancels_inflight_pipeline() -> None:
    outcome: dict[str, object] = {}

    class SlowPipeline:
        async def handle(self, runtime: SessionRuntime, context: TurnContext) -> None:
            try:
                await asyncio.sleep(30)
                outcome["finished"] = True
            except asyncio.CancelledError:
                outcome["cancelled"] = True
                raise

    runtime = SessionRuntime("s1", "c1", make_settings(), providers=make_noop_providers())
    runtime.pipeline = SlowPipeline()
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")
    runtime.attach("owner-a")

    await runtime.ingest_audio(_pcm(tone_samples(0.2, amplitude=0.3)))
    await runtime.bus.publish_and_await(TranscriptPartial(session_id="s1", turn_id=None, text="check my balance"))
    await runtime.ingest_audio(_pcm(silence_samples(0.3)))

    await _wait_until(lambda: bool(_events_of(collector, EventType.TURN_STARTED)))
    assert runtime.state is RuntimeState.PROCESSING

    await runtime.ingest_audio(_pcm(tone_samples(0.2, amplitude=0.3)))
    assert runtime.state is RuntimeState.INTERRUPTED

    await _wait_until(lambda: outcome.get("cancelled") is True)
    assert outcome.get("finished") is None

    completed = _events_of(collector, EventType.TURN_COMPLETED)
    assert completed and completed[-1].outcome == "cancelled"
    await runtime.detach("owner-a")


async def test_disconnect_cancels_inflight_work() -> None:
    outcome: dict[str, object] = {}

    class SlowPipeline:
        async def handle(self, runtime: SessionRuntime, context: TurnContext) -> None:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                outcome["cancelled"] = True
                raise

    runtime = SessionRuntime("s1", "c1", make_settings(), providers=make_noop_providers())
    runtime.pipeline = SlowPipeline()
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")
    runtime.attach("owner-a")

    await runtime.ingest_audio(_pcm(tone_samples(0.2, amplitude=0.3)))
    await runtime.bus.publish_and_await(TranscriptPartial(session_id="s1", turn_id=None, text="please help"))
    await runtime.ingest_audio(_pcm(silence_samples(0.3)))

    await _wait_until(lambda: bool(_events_of(collector, EventType.TURN_STARTED)))
    await runtime.detach("owner-a")

    await _wait_until(lambda: outcome.get("cancelled") is True)
    assert outcome.get("cancelled") is True
    assert runtime.state is RuntimeState.CLOSED
    assert runtime._active_scope is None


async def test_stale_transcript_after_close_is_ignored() -> None:
    runtime = SessionRuntime("s1", "c1", make_settings())
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")
    runtime.attach("owner-a")
    await runtime.detach("owner-a")
    await runtime.ingest_audio(_pcm(tone_samples(0.2, amplitude=0.3)))
    await runtime.bus.publish_and_await(TranscriptPartial(session_id="s1", turn_id=None, text="late audio"))
    assert runtime.turn_count == 0


async def test_snapshot_exposes_runtime_state() -> None:
    runtime = SessionRuntime("s1", "c1", make_settings())
    runtime.attach("owner-a")
    snap = runtime.snapshot()
    assert snap["session_id"] == "s1"
    assert snap["conversation_id"] == "c1"
    assert snap["state"] == "listening"
    assert snap["audio_connected"] is True
    assert snap["turn_count"] == 0
    await runtime.detach("owner-a")


async def test_vad_events_published_in_order() -> None:
    runtime = SessionRuntime("s1", "c1", make_settings())
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")
    runtime.attach("owner-a")

    await runtime.ingest_audio(_pcm(tone_samples(0.2, amplitude=0.3)))
    await runtime.ingest_audio(_pcm(silence_samples(0.3)))

    await _wait_until(
        lambda: len([e for e in collector if e.type in (EventType.SPEECH_STARTED, EventType.SPEECH_ENDED)]) == 2
    )
    kinds = [e.type for e in collector if e.type in (EventType.SPEECH_STARTED, EventType.SPEECH_ENDED)]
    assert kinds == [EventType.SPEECH_STARTED, EventType.SPEECH_ENDED]
    await runtime.detach("owner-a")
