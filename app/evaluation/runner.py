from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.evaluation.scenarios import (
    failing_registry,
    scenario_ambiguous_speech,
    scenario_backchannel,
    scenario_interruption,
    scenario_network_degradation,
    scenario_simple_question,
    scenario_tool_call,
    scenario_tool_failure,
)
from app.evaluation.support import SessionHarness

ScenarioHandler = Callable[[SessionHarness], Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class ScenarioSpec:
    name: str
    handler: ScenarioHandler
    overrides: dict[str, object] = field(default_factory=dict)
    slow_tts: bool = False
    tools: Callable[[], Any] | None = None


SCENARIOS: list[ScenarioSpec] = [
    ScenarioSpec(name="simple_question", handler=scenario_simple_question),
    ScenarioSpec(name="tool_call", handler=scenario_tool_call),
    ScenarioSpec(name="interruption", handler=scenario_interruption, slow_tts=True),
    ScenarioSpec(name="backchannel", handler=scenario_backchannel, slow_tts=True),
    ScenarioSpec(
        name="ambiguous_speech",
        handler=scenario_ambiguous_speech,
        overrides={"turn_silence_ms": 900},
    ),
    ScenarioSpec(name="tool_failure", handler=scenario_tool_failure, tools=failing_registry),
    ScenarioSpec(name="network_degradation", handler=scenario_network_degradation),
]

_BY_NAME = {spec.name: spec for spec in SCENARIOS}


async def run_scenario(name: str) -> dict[str, Any]:
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unknown scenario {name!r}; available: {sorted(_BY_NAME)}")
    tools = spec.tools() if spec.tools is not None else None
    harness = SessionHarness(settings_overrides=spec.overrides, slow_tts=spec.slow_tts, tools=tools)
    started = time.monotonic()
    await harness.open()
    try:
        details: dict[str, Any] = {}
        try:
            details = await spec.handler(harness)
        except Exception as exc:  # pragma: no cover - defensive
            details = {"checks": {}, "error": f"{type(exc).__name__}: {exc}"}
        checks = details.get("checks") or {}
        passed = bool(checks) and all(checks.values())
        latency = harness.latencies()
        counters = harness.counters()
        summary_latencies = {k: v.get("median") for k, v in latency.items() if v.get("median") is not None}
        return {
            "name": name,
            "passed": passed,
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
            "checks": checks,
            "counters": counters,
            "latencies_ms": summary_latencies,
            **{k: v for k, v in details.items() if k != "checks"},
        }
    finally:
        await harness.close()


async def run_all() -> dict[str, Any]:
    run_id = uuid.uuid4().hex
    started = time.time()
    results = [await run_scenario(spec.name) for spec in SCENARIOS]
    passed = sum(1 for r in results if r["passed"])
    return {
        "run_id": run_id,
        "started_at": started,
        "duration_ms": round((time.time() - started) * 1000, 2),
        "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed},
        "scenarios": results,
    }


def list_scenario_names() -> list[str]:
    return list(_BY_NAME)
