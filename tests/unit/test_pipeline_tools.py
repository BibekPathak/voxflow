from __future__ import annotations

import asyncio

from app.audio.resampling import float32_to_pcm16_bytes
from app.memory.conversation import ConversationStore
from app.providers.factory import ProviderSet
from app.providers.llm.mock import MockLLMProvider
from app.providers.tts.mock import MockTTSProvider
from app.runtime.errors import ToolError
from app.runtime.events import EventType, TranscriptPartial, VoiceEvent
from app.runtime.orchestrator import SessionRuntime, TurnContext
from app.runtime.state_machine import RuntimeState
from app.tools.registry import ToolRegistry
from tests.conftest import make_noop_providers, make_settings
from tests.unit.test_vad import silence_samples, tone_samples


def _pcm(samples) -> bytes:
    return float32_to_pcm16_bytes(samples)


async def _wait_until(condition, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


def _events_of(collector: list[VoiceEvent], event_type: EventType) -> list[VoiceEvent]:
    return [e for e in collector if e.type is event_type]


def _stt_free_providers() -> ProviderSet:
    noop = make_noop_providers()
    return ProviderSet(stt=noop.stt, llm=MockLLMProvider(), tts=MockTTSProvider())


async def _start_turn(runtime: SessionRuntime, text: str) -> None:
    frame = 320
    tone = tone_samples(0.2, amplitude=0.3)
    for i in range(0, len(tone), frame):
        await runtime.ingest_audio(_pcm(tone[i : i + frame]))
    await runtime.bus.publish_and_await(TranscriptPartial(session_id="s1", text=text))
    silence = silence_samples(0.35)
    for i in range(0, len(silence), frame):
        await runtime.ingest_audio(_pcm(silence[i : i + frame]))


async def test_tool_call_executes_and_second_pass_answers() -> None:
    runtime = SessionRuntime("s1", "c1", make_settings(), providers=_stt_free_providers())
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")
    runtime.attach("owner-a")

    await _start_turn(runtime, "can you inspect payment pay_103 for me")

    await _wait_until(lambda: bool(_events_of(collector, EventType.TURN_COMPLETED)))
    completed = _events_of(collector, EventType.TURN_COMPLETED)[-1]
    assert completed.outcome == "completed"

    starts = _events_of(collector, EventType.TOOL_CALL_STARTED)
    assert starts and starts[0].tool_name == "inspect_payment"
    assert starts[0].arguments == '{"payment_id": "pay_103"}'

    done = _events_of(collector, EventType.TOOL_CALL_COMPLETED)
    assert done and done[0].tool_name == "inspect_payment"
    assert isinstance(done[0].result, dict)
    assert done[0].result["status"] == "declined"
    assert done[0].result["decline_reason"] == "card_expired"

    spoken = "".join(e.text for e in _events_of(collector, EventType.LLM_TOKEN))
    assert "card_expired" in spoken

    assert len(runtime.history) == 2
    assert runtime.history[0].role == "user"
    assert runtime.history[1].role == "assistant"
    assert runtime.state is RuntimeState.LISTENING
    await runtime.detach("owner-a")


async def test_tool_failure_recovers_gracefully() -> None:
    registry = ToolRegistry()

    async def broken(payment_id: str) -> None:
        raise ToolError("inspect_payment", "backend unavailable")

    registry.register_tool(broken, name="inspect_payment")

    runtime = SessionRuntime("s1", "c1", make_settings(), providers=_stt_free_providers(), tools=registry)
    collector: list[VoiceEvent] = []
    runtime.bus.subscribe(lambda e: collector.append(e), session_id="s1")
    runtime.attach("owner-a")

    await _start_turn(runtime, "please inspect the payment pay_101")

    await _wait_until(lambda: bool(_events_of(collector, EventType.TURN_COMPLETED)))
    completed = _events_of(collector, EventType.TURN_COMPLETED)[-1]
    assert completed.outcome == "completed"

    failed = _events_of(collector, EventType.TOOL_CALL_FAILED)
    assert failed and failed[0].tool_name == "inspect_payment"
    assert "backend unavailable" in failed[0].error

    spoken = "".join(e.text for e in _events_of(collector, EventType.LLM_TOKEN))
    assert "backend unavailable" in spoken
    assert runtime.state is RuntimeState.LISTENING
    await runtime.detach("owner-a")


async def test_completed_turns_persist_to_conversation_store() -> None:
    store = ConversationStore("sqlite+aiosqlite:///:memory:")
    await store.create_tables()
    runtime = SessionRuntime("s1", "c1", make_settings(), providers=make_noop_providers(), conversation_store=store)
    context = TurnContext(session_id="s1", conversation_id="c1", turn_id=1, text="my payment failed")
    context.response_text = "let me look into that"
    await runtime._commit_turn(context)

    rows = await store.recent("c1")
    assert [(r["role"], r["content"]) for r in rows] == [
        ("user", "my payment failed"),
        ("assistant", "let me look into that"),
    ]
    assert len(runtime.history) == 2
    await store.close()
