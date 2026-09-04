from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.evaluation.support import wait_until
from app.providers.factory import build_providers, provider_fingerprint
from app.runtime.events import EventType, VoiceEvent
from app.runtime.orchestrator import SessionRuntime

# ---------------------------------------------------------------------------
# Default benchmark script: support-style prompts that should trigger tools.
# ---------------------------------------------------------------------------
DEFAULT_SCRIPT: list[str] = [
    "what fees do you charge",
    "can you inspect payment pay_101 for me",
    "why was my payment declined",
    "search for the customer alice",
    "list my recent transactions",
    "please open a support ticket about a failed payment",
    "who is the customer bob",
    "show me the most recent payments for customer cust 2",
]


@dataclass(slots=True)
class BenchTurnResult:
    turn_id: int
    prompt: str
    outcome: str
    correct: bool | None
    ttft_ms: float | None
    ttfa_ms: float | None
    e2e_ms: float | None
    tts_request_ms: float | None
    started_at: float
    finished_at: float


@dataclass(slots=True)
class BenchReport:
    run_id: str
    started_at: float
    duration_ms: float
    turns: int
    passed: int
    fingerprint: dict[str, object]
    provider_description: dict[str, dict[str, object] | None]
    baseline_mock: dict[str, float] | None
    results: list[BenchTurnResult] = field(default_factory=list)


_PCM_DIR = Path(__file__).resolve().parent / ".." / ".." / "eval_audio"
_REPORT_DIR = Path(__file__).resolve().parent / ".." / ".." / "eval_reports"


def _collect(results: list[BenchTurnResult], key: str) -> list[float]:
    return [getattr(r, key) for r in results if getattr(r, key) is not None]


def _frame_data(samples: Any) -> bytes:
    from app.audio.resampling import float32_to_pcm16_bytes

    return (
        float32_to_pcm16_bytes(samples)
        if hasattr(samples, "tobytes")
        else float32_to_pcm16_bytes(__import__("numpy").asarray(samples, dtype="float32"))
    )


def _frames_pcm(pcm: bytes, frame_size_bytes: int = 320) -> list[bytes]:
    return [pcm[i : i + frame_size_bytes] for i in range(0, len(pcm), frame_size_bytes)]


def _cache_path(provider_key: str, index: int, prompt: str, sample_rate: int) -> Path:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    return _PCM_DIR / f"{provider_key}_p{index}_{digest}_{sample_rate}.json"


async def _generate_prompt_audio(tts: Any, prompt: str, index: int, sample_rate: int, *, force: bool = False) -> bytes:
    """Synthesize a spoken prompt via the configured TTS provider and cache it.

    Caching keeps repeated benchmark runs from re-hitting the provider and
    keeps STT/LLM/TTS measurement isolated from prompt generation.
    """
    provider_name = tts.metadata().name if hasattr(tts, "metadata") else "tts"
    path = _cache_path(provider_name, index, prompt, sample_rate)
    if path.exists() and not force:
        payload = json.loads(path.read_text())
        return bytes(payload["pcm"])
    audio = await tts.synthesize(prompt, sample_rate=sample_rate)
    _PCM_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sample_rate": audio.sample_rate, "pcm": list(audio.pcm)}))
    return audio.pcm


async def _run_single_turn(
    runtime: SessionRuntime,
    prompt_pcm: bytes,
    sample_rate: int,
    *,
    frame_size_bytes: int = 320,
) -> BenchTurnResult:
    turn_id = runtime.turn_count + 1
    started = time.monotonic()
    events: list[VoiceEvent] = []
    sub = runtime.bus.subscribe(events.append, session_id=runtime.session_id)

    speech = _frames_pcm(prompt_pcm, frame_size_bytes)
    silence = bytes(frame_size_bytes)
    try:
        for chunk in speech:
            await runtime.ingest_audio(chunk)
        for _ in range(int((sample_rate * 0.6) // frame_size_bytes)):
            await runtime.ingest_audio(silence)
        await wait_until(
            lambda: any(e.type is EventType.TURN_COMPLETED and e.turn_id == turn_id for e in events),
            timeout=30.0,
        )
    finally:
        sub.close()

    finished = time.monotonic()
    tool_starts = [e for e in events if e.type is EventType.TOOL_CALL_STARTED and e.turn_id == turn_id]
    transcript_finals = [e.text for e in events if e.type is EventType.TRANSCRIPT_FINAL and e.turn_id == turn_id]
    outcome_event = next((e for e in events if e.type is EventType.TURN_COMPLETED and e.turn_id == turn_id), None)
    outcome = getattr(outcome_event, "outcome", "error") if outcome_event else "error"
    spoken = transcript_finals[-1] if transcript_finals else ""

    turn_rows = [t for t in runtime.metrics._completed_turns if t.get("turn_id") == turn_id]
    row = turn_rows[0] if turn_rows else {}
    return BenchTurnResult(
        turn_id=turn_id,
        prompt=spoken,
        outcome=outcome,
        correct=_verdict(prompt_text=spoken, tool_starts=tool_starts),
        ttft_ms=row.get("ttft_ms"),
        ttfa_ms=row.get("ttfa_ms"),
        e2e_ms=row.get("e2e_ms"),
        tts_request_ms=row.get("tts_request_ms"),
        started_at=started,
        finished_at=finished,
    )


def _verdict(*, prompt_text: str, tool_starts: list[VoiceEvent]) -> bool | None:
    lowered = prompt_text.lower()
    if "payment" in lowered or "transaction" in lowered or "declined" in lowered:
        return any(getattr(e, "tool_name", "") == "inspect_payment" for e in tool_starts)
    if "ticket" in lowered:
        return any(getattr(e, "tool_name", "") == "create_support_ticket" for e in tool_starts)
    if "customer" in lowered:
        return any(getattr(e, "tool_name", "") == "search_customer" for e in tool_starts)
    return None


async def run_benchmark(
    *,
    turns: int = 20,
    script: list[str] | None = None,
    settings: Settings | None = None,
    tts_override: Any | None = None,
    force_generate: bool = False,
    sample_rate: int = 16_000,
    baseline_mock: dict[str, float] | None = None,
    runtime_factory: Callable[..., Awaitable[SessionRuntime]] | None = None,
    close_runtime: Callable[..., Awaitable[None]] | None = None,
) -> BenchReport:
    """Run ``turns`` full voice conversations through the configured providers.

    If ``runtime_factory`` / ``close_runtime`` are provided they are used so
    callers can inject a custom runtime; otherwise a default runtime is built
    from ``build_providers(settings)`` (real providers when configured).
    """
    prompts = script or DEFAULT_SCRIPT
    report = BenchReport(
        run_id=f"bench_{uuid.uuid4().hex[:12]}",
        started_at=time.time(),
        duration_ms=0.0,
        turns=turns,
        passed=0,
        fingerprint=provider_fingerprint(settings) if settings else {},
        provider_description={"stt": None, "llm": None, "tts": None},
        baseline_mock=baseline_mock,
    )
    started = time.monotonic()

    own_runtime = runtime_factory is None
    if own_runtime:
        if settings is None:
            raise ValueError("settings is required when no runtime_factory is provided")
        providers = build_providers(settings)
        runtime = SessionRuntime(
            session_id=f"bench_{uuid.uuid4().hex[:12]}",
            conversation_id=f"conv_{uuid.uuid4().hex[:12]}",
            settings=settings,
            providers=providers,
        )
        report.provider_description = providers.describe()
        runtime.attach("bench")
    else:
        runtime = await runtime_factory()
        report.provider_description = runtime.providers.describe()

    passed = 0
    try:
        for index in range(turns):
            prompt = prompts[index % len(prompts)]
            tts = tts_override if tts_override is not None else runtime.providers.tts
            prompt_pcm = await _generate_prompt_audio(tts, prompt, index, sample_rate, force=force_generate)
            result = await _run_single_turn(runtime, prompt_pcm, sample_rate)
            if result.outcome == "completed" and result.correct is not False:
                passed += 1
            report.results.append(result)
    finally:
        if own_runtime:
            await runtime.detach("bench")
            await runtime.providers.close()
        elif close_runtime is not None:
            await close_runtime(runtime)

    report.duration_ms = (time.monotonic() - started) * 1000
    report.passed = passed
    return report


# ---------------------------------------------------------------------------
# Report serialization (JSON + Markdown).
# ---------------------------------------------------------------------------
async def run_mock_baseline(*, turns: int = 12) -> dict[str, float]:
    """Measure a mock-provider baseline (median TTFT/TTFA/E2E) for comparison.

    Uses the scenario harness path (injected transcripts -> mock LLM -> mock
    TTS) which is deterministic and matches the measured numbers already in the
    README, so the Mock vs Real comparison is apples-to-apples on the runtime.
    """
    from app.evaluation.support import SessionHarness
    from app.observability.metrics import latency_aggregate

    harness = SessionHarness()
    await harness.open()
    script = DEFAULT_SCRIPT
    medians: dict[str, list[float]] = {"ttft": [], "ttfa": [], "e2e": []}
    try:
        for index in range(turns):
            await harness.say(script[index % len(script)])
            await harness.wait_turns(index + 1)
            latency = harness.latencies()
            for key in medians:
                value = latency.get(key, {}).get("median")
                if value is not None:
                    medians[key].append(float(value))
    finally:
        await harness.close()

    def median_of(values: list[float]) -> float:
        return latency_aggregate(values)["median"] or 0.0

    return {"ttft": median_of(medians["ttft"]), "ttfa": median_of(medians["ttfa"]), "e2e": median_of(medians["e2e"])}


def _report_to_dict(report: BenchReport) -> dict[str, Any]:
    from app.observability.metrics import latency_aggregate

    metric_rows: dict[str, dict[str, float | int | None]] = {}
    means: dict[str, float] = {}
    for key in ("ttft_ms", "ttfa_ms", "e2e_ms", "tts_request_ms"):
        values = _collect(report.results, key)
        metric_rows[key.rstrip("_ms")] = latency_aggregate(values)
        if values:
            means[key.rstrip("_ms")] = round(sum(values) / len(values), 2)
    return {
        "run_id": report.run_id,
        "started_at": report.started_at,
        "duration_ms": round(report.duration_ms, 2),
        "turns": report.turns,
        "completed": sum(1 for r in report.results if r.outcome == "completed"),
        "correct": sum(1 for r in report.results if r.correct is True),
        "passed": report.passed,
        "provider_fingerprint": report.fingerprint,
        "providers": report.provider_description,
        "baseline_mock": report.baseline_mock,
        "latency_means_ms": means,
        "latency_p": {k: {"median": v["median"], "p95": v.get("p95")} for k, v in metric_rows.items()},
        "results": [
            {
                "turn_id": r.turn_id,
                "prompt": r.prompt,
                "outcome": r.outcome,
                "correct": r.correct,
                "ttft_ms": r.ttft_ms,
                "ttfa_ms": r.ttfa_ms,
                "e2e_ms": r.e2e_ms,
                "tts_request_ms": r.tts_request_ms,
            }
            for r in report.results
        ],
    }


def bench_report_markdown(report: BenchReport) -> str:
    from app.observability.metrics import latency_aggregate

    lines = [
        "# VoxFlow Real Provider Benchmark",
        "",
        f"- Run: `{report.run_id}`",
        f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(report.started_at))}",
        f"- Duration: {report.duration_ms:.0f} ms",
        f"- Turns: {report.turns} (completed {sum(1 for r in report.results if r.outcome == 'completed')}, "
        f"correct {sum(1 for r in report.results if r.correct is True)})",
        f"- Passed: {report.passed}/{report.turns}",
        "",
        "## Providers",
        "",
        "| Stage | Provider | Vendor | Model | Streaming |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stage in ("stt", "llm", "tts"):
        info = report.provider_description.get(stage)
        if info:
            lines.append(
                f"| {stage} | {info.get('name')} | {info.get('vendor')} | {info.get('model')} | "
                f"{bool(info.get('streaming'))} |"
            )
    lines += ["", "## Configuration fingerprint", "", "```json", json.dumps(report.fingerprint, indent=2), "```"]

    if report.baseline_mock:
        lines += ["", "## Mock vs Real (median, ms)", "", "| Metric | Mock | Real |", "| --- | --- | --- |"]
        for key, display in (("ttft", "TTFT"), ("ttfa", "TTFA"), ("e2e", "E2E")):
            real = _median_for(report, key)
            mock_value = report.baseline_mock.get(key)
            lines.append(
                f"| {display} | {'—' if mock_value is None else round(mock_value)} | "
                f"{'—' if real is None else round(real)} |"
            )
    lines += ["", "## Latency percentiles (ms)", "", "| Metric | P50 | P95 |", "| --- | --- | --- |"]
    for key, display in (("ttft", "TTFT"), ("ttfa", "TTFA"), ("e2e", "E2E"), ("tts_request", "TTS request")):
        agg = latency_aggregate(_collect(report.results, f"{key}_ms"))
        lines.append(f"| {display} | {agg['median']} | {agg.get('p95')} |")
    header = "| Turn | Outcome | Correct | TTFT | TTFA | E2E |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    lines += ["", "## Per-turn results", "", header, sep]
    for r in report.results:
        lines.append(
            f"| {r.turn_id} | {r.outcome} | {r.correct} | {_fmt(r.ttft_ms)} | {_fmt(r.ttfa_ms)} | {_fmt(r.e2e_ms)} |"
        )
    return "\n".join(lines) + "\n"


def _median_for(report: BenchReport, key: str) -> float | None:
    from app.observability.metrics import latency_aggregate

    return latency_aggregate(_collect(report.results, f"{key}_ms"))["median"]


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def save_report(report: BenchReport, *, directory: Path | None = None) -> Path:
    target = directory or _REPORT_DIR
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{report.run_id}.json"
    json_path.write_text(json.dumps(_report_to_dict(report), indent=2))
    return json_path


def save_report_markdown(report: BenchReport, directory: Path | None = None) -> Path:
    target = directory or _REPORT_DIR
    target.mkdir(parents=True, exist_ok=True)
    md_path = target / f"{report.run_id}.md"
    md_path.write_text(bench_report_markdown(report))
    return md_path
