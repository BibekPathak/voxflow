from __future__ import annotations

import time

from app.evaluation.support import SessionHarness
from app.runtime.errors import ToolError
from app.runtime.events import EventType
from app.runtime.state_machine import RuntimeState
from app.tools.registry import ToolRegistry


def _completed_outcome(harness: SessionHarness) -> str | None:
    completed = harness.last_completed()
    return completed.outcome if completed else None


def _assistant_text(harness: SessionHarness) -> str:
    assistants = [entry for entry in harness.runtime.history if entry.role == "assistant"]
    return assistants[-1].content if assistants else ""


async def scenario_simple_question(harness: SessionHarness) -> dict:
    await harness.say("what fees do you charge")
    reached = await harness.wait_turns(1)
    outcome = _completed_outcome(harness)
    text = _assistant_text(harness)
    tts_frames = harness.of_type(EventType.TTS_AUDIO)
    latency = harness.latencies()
    checks = {
        "turn_completed": reached and outcome == "completed",
        "agent_speaks": len(tts_frames) > 0,
        "responds_in_text": len(text) > 0 and "help" in text.lower(),
        "ttft_measured": latency.get("ttft", {}).get("median") is not None,
        "ttfa_measured": latency.get("ttfa", {}).get("median") is not None,
    }
    return {"checks": checks, "response": text, "latencies": {k: v.get("median") for k, v in latency.items()}}


async def scenario_tool_call(harness: SessionHarness) -> dict:
    await harness.say("can you inspect payment pay_101 for me")
    reached = await harness.wait_turns(1)
    outcome = _completed_outcome(harness)

    starts = harness.of_type(EventType.TOOL_CALL_STARTED)
    finished = harness.of_type(EventType.TOOL_CALL_COMPLETED)
    selected = starts[0].tool_name if starts else None
    args = starts[0].arguments if starts else ""
    result_ok = bool(finished) and isinstance(finished[0].result, dict) and finished[0].result.get("found") is True
    status = (finished[0].result or {}).get("status") if finished else None
    text = _assistant_text(harness)

    checks = {
        "turn_completed": reached and outcome == "completed",
        "tool_selected": selected == "inspect_payment",
        "tool_arguments_correct": "pay_101" in args,
        "tool_result_correct": result_ok and status == "declined",
        "final_response_correct": "insufficient_funds" in text,
    }
    return {"checks": checks, "tool": selected, "arguments": args, "response": text}


async def scenario_interruption(harness: SessionHarness) -> dict:
    await harness.say("please explain how refunds work", tone_seconds=0.2, silence_seconds=0.4)
    spoke = await _wait_until_speaking(harness)
    if not spoke:
        return {"checks": {"agent_starts_speaking": False}}

    await harness.burst(0.3)
    await harness.silence_pause(0.5)

    interrupted_seen = await _wait_until_event_count(harness, EventType.USER_INTERRUPTED, 1, 3.0)
    cancelled_seen = await _wait_until_outcome_cancelled(harness, 3.0)

    frames_before = len(harness.outbound.audio)
    await _sleep(0.3)
    frames_after = len(harness.outbound.audio)
    stale_frames = frames_after > frames_before

    metrics = harness.latencies()
    frames_after_cancel = len(harness.outbound.audio)
    await harness.say("thank you that helps", tone_seconds=0.2, silence_seconds=0.4)
    second_started = await _wait_until_event_count(harness, EventType.TURN_STARTED, 2, 5.0)
    second_speaking = await _wait_until_speaking_after(harness, frames_after_cancel)

    checks = {
        "agent_starts_speaking": spoke,
        "interrupt_detected": interrupted_seen,
        "turn_cancelled": cancelled_seen,
        "no_stale_audio_after_cancel": not stale_frames,
        "interruption_latency_measured": metrics.get("interruption", {}).get("median") is not None,
        "cancellation_latency_measured": metrics.get("tts_cancellation", {}).get("median") is not None,
        "new_turn_processed": second_started and second_speaking,
    }
    return {"checks": checks, "latencies": {k: v.get("median") for k, v in metrics.items()}}


async def scenario_backchannel(harness: SessionHarness) -> dict:
    await harness.say("what is the refund policy", tone_seconds=0.2, silence_seconds=0.4)
    spoke = await _wait_until_speaking(harness)
    if not spoke:
        return {"checks": {"agent_starts_speaking": False}}

    await harness.burst(0.03, amplitude=0.4)
    frames_before = len(harness.outbound.audio)
    await _sleep(0.25)
    frames_after = len(harness.outbound.audio)

    checks = {
        "agent_starts_speaking": spoke,
        "no_interruption": len(harness.of_type(EventType.USER_INTERRUPTED)) == 0,
        "agent_keeps_talking": frames_after > frames_before,
        "no_restart": len(harness.of_type(EventType.TURN_STARTED)) == 1,
    }
    return {"checks": checks}


async def scenario_ambiguous_speech(harness: SessionHarness) -> dict:
    await harness.burst(0.15)
    await harness.runtime.bus.publish_and_await(_partial(harness, "I need to change my"))
    await harness.silence_pause(0.35)

    await harness.burst(0.2)
    await harness.runtime.bus.publish_and_await(_partial(harness, "I need to change my billing address"))
    await harness.silence_pause(1.2)

    reached = await harness.wait_turns(1, timeout=5.0)
    outcome = _completed_outcome(harness)
    turns = len(harness.of_type(EventType.TURN_STARTED))
    user_entries = [entry.content for entry in harness.runtime.history if entry.role == "user"]

    checks = {
        "no_premature_endpointing": turns == 1,
        "turn_completed": reached and outcome == "completed",
        "full_utterance_captured": bool(user_entries)
        and user_entries[-1].strip() == "I need to change my billing address",
    }
    return {"checks": checks, "utterance": user_entries[-1] if user_entries else None}


async def scenario_tool_failure(harness: SessionHarness) -> dict:
    await harness.say("please inspect the payment pay_101")
    reached = await harness.wait_turns(1)
    outcome = _completed_outcome(harness)
    failed = harness.of_type(EventType.TOOL_CALL_FAILED)
    text = _assistant_text(harness)
    checks = {
        "turn_recovers": reached and outcome == "completed",
        "tool_failure_observed": bool(failed) and failed[0].tool_name == "inspect_payment",
        "graceful_response": "backend unavailable" in text or "unavailable" in text,
        "session_listening_again": harness.state is RuntimeState.LISTENING,
    }
    return {"checks": checks, "response": text}


async def scenario_network_degradation(harness: SessionHarness) -> dict:
    sequence = 0
    for chunk in _frames_of(0.35):
        if sequence % 7 in (3, 4):
            sequence += 1
            continue
        await harness.feed(chunk, sequence=sequence)
        sequence += 1
    await harness.runtime.bus.publish_and_await(_partial(harness, "check my recent payments"))
    for chunk in _silence_frames_of(0.6):
        await harness.feed(chunk, sequence=sequence)
        sequence += 1

    reached = await harness.wait_turns(1)
    outcome = _completed_outcome(harness)
    gaps = harness.runtime.gateway.sequencer.total_gaps
    missing = harness.runtime.gateway.sequencer.missing_frames
    checks = {
        "dropped_frames_detected": gaps >= 1 and missing >= 1,
        "turn_still_completes": reached and outcome == "completed",
        "session_listening_again": harness.state is RuntimeState.LISTENING,
    }
    return {"checks": checks, "dropped": missing}


def _frames_of(seconds: float):
    from app.evaluation.support import frames, tone

    return frames(tone(seconds, amplitude=0.3))


def _silence_frames_of(seconds: float):
    from app.evaluation.support import frames, silence

    return frames(silence(seconds))


def _partial(harness: SessionHarness, text: str):
    from app.runtime.events import TranscriptPartial

    return TranscriptPartial(session_id=harness.runtime.session_id, text=text, timestamp=time.time())


async def _wait_until_speaking(harness: SessionHarness, timeout: float = 5.0) -> bool:
    from app.evaluation.support import wait_until

    return await wait_until(lambda: len(harness.outbound.audio) > 0, timeout)


async def _wait_until_event_count(harness: SessionHarness, event_type: EventType, count: int, timeout: float) -> bool:
    from app.evaluation.support import wait_until

    return await wait_until(lambda: len(harness.of_type(event_type)) >= count, timeout)


async def _wait_until_speaking_after(harness: SessionHarness, minimum_frames: int, timeout: float = 5.0) -> bool:
    from app.evaluation.support import wait_until

    return await wait_until(lambda: len(harness.outbound.audio) > minimum_frames, timeout)


async def _wait_until_outcome_cancelled(harness: SessionHarness, timeout: float) -> bool:
    from app.evaluation.support import wait_until

    return await wait_until(
        lambda: bool([e for e in harness.of_type(EventType.TURN_COMPLETED) if e.outcome == "cancelled"]),
        timeout,
    )


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def failing_registry() -> ToolRegistry:
    registry = ToolRegistry()

    async def broken(payment_id: str) -> None:
        raise ToolError("inspect_payment", "backend unavailable")

    registry.register_tool(broken, name="inspect_payment")
    return registry
