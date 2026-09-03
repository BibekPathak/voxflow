from __future__ import annotations

import asyncio
import time

import pytest

from app.audio.resampling import float32_to_pcm16_bytes
from app.observability.metrics import MetricsCollector
from app.runtime.events import (
    LLMStarted,
    LLMToken,
    SpeechStarted,
    TranscriptFinal,
    TranscriptPartial,
    TTSAudio,
    TTSCompleted,
    TTSStarted,
    TurnCompleted,
    UserInterrupted,
)
from app.runtime.orchestrator import SessionRuntime
from tests.conftest import make_settings
from tests.unit.test_vad import silence_samples, tone_samples


def _pcm(samples) -> bytes:
    return float32_to_pcm16_bytes(samples)


async def _wait_until(condition, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


def test_turn_latency_ledger_computes_markers() -> None:
    collector = MetricsCollector()
    base = time.time()
    collector.register_turn(1, speech_end_ts=base - 0.4)
    collector.on_event(LLMStarted(session_id="s", turn_id=1, timestamp=base + 0.05))
    collector.on_event(LLMToken(session_id="s", turn_id=1, timestamp=base + 0.1, text="hi"))
    collector.on_event(TTSStarted(session_id="s", turn_id=1, timestamp=base + 0.15))
    collector.on_event(TTSAudio(session_id="s", turn_id=1, timestamp=base + 0.2))
    collector.on_event(TurnCompleted(session_id="s", turn_id=1, timestamp=base + 0.5, outcome="completed"))

    snap = collector.snapshot()
    assert snap["counters"]["turns_completed"] == 1
    assert snap["counters"]["outcome_completed"] == 1
    ttft = snap["latencies_ms"]["ttft"]["median"]
    ttfa = snap["latencies_ms"]["ttfa"]["median"]
    e2e = snap["latencies_ms"]["e2e"]["median"]
    assert ttft is not None and 90 <= ttft <= 110
    assert ttfa is not None and 190 <= ttfa <= 210
    assert e2e is not None and 590 <= e2e <= 610


def test_interruption_and_cancellation_latencies() -> None:
    collector = MetricsCollector()
    base = time.time()
    collector.register_turn(7, speech_end_ts=base - 0.3)
    collector.on_event(SpeechStarted(session_id="s", turn_id=7, timestamp=base))
    collector.on_event(UserInterrupted(session_id="s", turn_id=7, interrupted_turn_id=7, timestamp=base + 0.04))
    collector.on_event(TTSCompleted(session_id="s", turn_id=7, timestamp=base + 0.15, reason="cancelled"))
    collector.on_event(TurnCompleted(session_id="s", turn_id=7, timestamp=base + 0.2, outcome="cancelled"))

    snap = collector.snapshot()
    assert snap["counters"]["user_interrupts"] == 1
    detection = snap["latencies_ms"]["interruption"]["median"]
    cancellation = snap["latencies_ms"]["tts_cancellation"]["median"]
    assert detection is not None and 35 <= detection <= 45
    assert cancellation is not None and 105 <= cancellation <= 115


def test_transcript_latencies_tracked_per_utterance() -> None:
    collector = MetricsCollector()
    base = time.time()
    collector.on_event(SpeechStarted(session_id="s", timestamp=base))
    collector.on_event(TranscriptPartial(session_id="s", timestamp=base + 0.4, text="hel"))
    collector.on_event(TranscriptFinal(session_id="s", timestamp=base + 0.9, text="hello"))

    snap = collector.snapshot()
    assert snap["transcript_ms"]["time_to_first_partial"]["median"] == pytest.approx(400, abs=10)
    assert snap["transcript_ms"]["time_to_final"]["median"] == pytest.approx(900, abs=10)


async def test_pipeline_completion_populates_metrics() -> None:
    runtime = SessionRuntime("s1", "c1", make_settings())
    runtime.attach("owner-a")
    frame = 320
    tone = tone_samples(0.5, amplitude=0.3)
    for i in range(0, len(tone), frame):
        await runtime.ingest_audio(_pcm(tone[i : i + frame]))
    silence = silence_samples(0.6)
    for i in range(0, len(silence), frame):
        await runtime.ingest_audio(_pcm(silence[i : i + frame]))

    await _wait_until(lambda: runtime.metrics.snapshot()["counters"].get("turns_completed", 0) >= 1)
    snap = runtime.metrics.snapshot()
    assert snap["counters"]["turns_completed"] >= 1
    assert snap["counters"]["outcome_completed"] >= 1
    assert snap["latencies_ms"]["ttft"]["median"] is not None
    assert snap["latencies_ms"]["ttfa"]["median"] is not None
    assert snap["transcript_ms"]["time_to_final"]["median"] is not None
    await runtime.detach("owner-a")
