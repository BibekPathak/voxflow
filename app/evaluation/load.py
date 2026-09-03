from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.support import SessionHarness, wait_until
from app.runtime.events import EventType


@dataclass(slots=True)
class LoadReport:
    concurrency: int = 0
    turns_per_session: int = 0
    sessions_completed: int = 0
    turns_completed: int = 0
    expected_turns: int = 0
    errors: int = 0
    stale_turns: int = 0
    elapsed_ms: float = 0.0
    per_session: list[dict[str, Any]] = field(default_factory=list)


def _summary(report: LoadReport) -> dict[str, Any]:
    return {
        "concurrency": report.concurrency,
        "turns_per_session": report.turns_per_session,
        "sessions_completed": report.sessions_completed,
        "turns_completed": report.turns_completed,
        "expected_turns": report.expected_turns,
        "errors": report.errors,
        "stale_turns": report.stale_turns,
        "elapsed_ms": round(report.elapsed_ms, 2),
        "turns_per_second": round(report.turns_completed / (report.elapsed_ms / 1000), 2)
        if report.elapsed_ms > 0
        else None,
    }


async def _drive_session(session_index: int, turns: int, marker: str) -> dict[str, Any]:
    harness = SessionHarness()
    await harness.open()
    started = time.monotonic()
    completed_turns = 0
    markers_observed: list[str] = []
    try:
        for turn in range(1, turns + 1):
            query = f"please inspect payment pay_101 for marker {session_index} turn {turn}"
            text = f"{marker} session {session_index}: {query}"
            await harness.say(text)
            reached = await wait_until(
                lambda turn=turn: len(harness.of_type(EventType.TURN_COMPLETED)) >= turn,
                timeout=8.0,
            )
            if reached:
                completed_turns += 1
            markers_observed.append(
                [entry.content for entry in harness.runtime.history if entry.role == "user"][-1]
                if harness.runtime.history
                else ""
            )
        counters = harness.counters()
        latency = harness.latencies()
        errors = counters.get("errors", 0)
        stale = 0
        for entry in markers_observed:
            if f"session {session_index}" not in entry:
                stale += 1
        return {
            "session_index": session_index,
            "completed_turns": completed_turns,
            "errors": errors,
            "stale_markers": stale,
            "state": harness.state.value,
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
            "counters": counters,
            "latencies_ms": {k: v.get("median") for k, v in latency.items() if v.get("median") is not None},
        }
    finally:
        await harness.close()


async def run_load_test(*, concurrency: int = 6, turns_per_session: int = 2) -> dict[str, Any]:
    """Runs several full voice conversations concurrently against isolated
    in-process runtimes (real VAD/endpointing/pipeline, mock providers)."""
    report = LoadReport(
        concurrency=concurrency,
        turns_per_session=turns_per_session,
        expected_turns=concurrency * turns_per_session,
    )
    started = time.monotonic()
    results = await asyncio.gather(
        *[_drive_session(index, turns_per_session, marker="cross-session-marker") for index in range(concurrency)]
    )
    report.elapsed_ms = (time.monotonic() - started) * 1000
    report.per_session = results
    report.sessions_completed = sum(1 for r in results if r["completed_turns"] == turns_per_session)
    report.turns_completed = sum(r["completed_turns"] for r in results)
    report.errors = sum(r["errors"] for r in results)
    report.stale_turns = sum(r["stale_markers"] for r in results)
    return _summary(report)


async def run_load_quick() -> dict[str, Any]:
    return await run_load_test(concurrency=4, turns_per_session=2)
