from __future__ import annotations

import statistics
import time
from collections import Counter
from typing import Any

from app.runtime.events import EventType, VoiceEvent

TURN_STAGE_MARKS = (
    EventType.LLM_STARTED,
    EventType.LLM_TOKEN,
    EventType.LLM_COMPLETED,
    EventType.TTS_STARTED,
    EventType.TTS_AUDIO,
    EventType.TTS_COMPLETED,
)


def _pct(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, int(percentile * len(sorted_values))))
    return sorted_values[index]


def latency_aggregate(
    values: list[float], *, percentiles: tuple[float, ...] = (50.0, 95.0)
) -> dict[str, float | int | None]:
    """Aggregate raw latency samples (ms) with median and arbitrary percentiles.

    Reused by both the per-session metrics snapshot and the benchmark harness so
    the P50/P95 definitions match everywhere.
    """
    clean = [v for v in values if v is not None and v >= 0]
    if not clean:
        return {"median": None, "p95": None, "n": 0}
    ordered = sorted(clean)
    result: dict[str, float | int | None] = {"median": round(statistics.median(clean), 2), "n": len(clean)}
    for percentile in percentiles:
        result[f"p{int(percentile):03d}"] = round(_pct(ordered, percentile / 100.0) or 0.0, 2)
    if "p095" in result:
        result["p95"] = result.pop("p095")
    return result


def _aggregate(values: list[float]) -> dict[str, float | None]:
    aggregated = latency_aggregate(values)
    return {"median": aggregated["median"], "p95": aggregated.get("p95"), "n": int(aggregated["n"])}


def _ms(since: float, now: float) -> float:
    return (now - since) * 1000.0


class MetricsCollector:
    """Per-session latency ledger and counters.

    Consumes the session event stream and records:

    * per-turn stage marks (LLM start/first token, TTS request/first audio,
      TTS stop) that yield TTFT, TTFA and end-to-end latency;
    * utterance-level transcript latency (VAD speech onset -> first partial /
      final transcript);
    * interruption latency (user speech onset -> INTERRUPTED) and TTS
      cancellation latency (interrupt request -> TTS actually stopped);
    * aggregate counters (turns, outcomes, tool calls, errors, interrupts).

    Latencies are computed in milliseconds from event timestamps and aggregated
    across completed turns with median / P95, so a session or evaluation run can
    answer "how slow was that conversation, and where did the time go?".
    """

    def __init__(self) -> None:
        self._active_turns: dict[int, dict[str, Any]] = {}
        self._completed_turns: list[dict[str, Any]] = []
        self._counters: Counter[str] = Counter()
        self._speech_started_ts: float | None = None
        self._partial_reported = False
        self._utterance_ttfp: list[float] = []
        self._utterance_ttf: list[float] = []

    def register_turn(self, turn_id: int, *, speech_end_ts: float | None = None) -> None:
        self._active_turns[turn_id] = {
            "turn_id": turn_id,
            "turn_start": time.time(),
            "speech_end": speech_end_ts,
            "marks": {},
            "interrupt_detected_ms": None,
            "interrupt_time": None,
            "tts_stop_ts": None,
            "outcome": None,
        }

    def _mark(self, turn_id: int, name: str, ts: float, *, first_only: bool = False) -> None:
        active = self._active_turns.get(turn_id)
        if active is None:
            return
        marks = active.setdefault("marks", {})
        if first_only and name in marks:
            return
        marks[name] = ts

    def _finalize_turn(self, turn_id: int) -> None:
        active = self._active_turns.pop(turn_id, None)
        if active is None:
            return
        marks = active["marks"]
        turn_start = active["turn_start"]
        speech_end = active["speech_end"]

        def delta(name: str, base: float | None = None) -> float | None:
            ts = marks.get(name)
            if ts is None:
                return None
            return round(_ms(base if base is not None else turn_start, ts), 2)

        ttfa = delta("tts_first_audio")
        record: dict[str, Any] = {
            "turn_id": turn_id,
            "outcome": active.get("outcome") or "completed",
            "ttft_ms": delta("llm_first_token"),
            "ttfa_ms": ttfa,
            "e2e_ms": (
                round(_ms(speech_end, marks["tts_first_audio"]), 2)
                if speech_end is not None and ttfa is not None
                else None
            ),
            "tts_request_ms": delta("tts_request"),
            "llm_start_to_first_token_ms": (
                round(_ms(marks.get("llm_start"), marks["llm_first_token"]), 2)
                if "llm_start" in marks and "llm_first_token" in marks
                else None
            ),
            "interrupt_detected_ms": active["interrupt_detected_ms"],
            "tts_cancellation_ms": (
                round(_ms(active["interrupt_time"], active["tts_stop_ts"]), 2)
                if active["interrupt_time"] is not None and active["tts_stop_ts"] is not None
                else None
            ),
        }
        self._completed_turns.append(record)

    def on_event(self, event: VoiceEvent) -> None:
        kind = event.type
        ts = event.timestamp
        turn_id = event.turn_id

        if kind is EventType.TURN_STARTED and turn_id is not None and turn_id not in self._active_turns:
            self.register_turn(turn_id)
        elif kind is EventType.TURN_COMPLETED and turn_id is not None:
            self._counters["turns_completed"] += 1
            if getattr(event, "outcome", None):
                self._counters[f"outcome_{event.outcome}"] += 1
            active = self._active_turns.get(turn_id)
            if active is not None:
                active["outcome"] = event.outcome
            self._finalize_turn(turn_id)

        if kind in TURN_STAGE_MARKS and turn_id is not None:
            if kind is EventType.TTS_AUDIO:
                self._mark(turn_id, "tts_first_audio", ts, first_only=True)
            elif kind is EventType.TTS_STARTED:
                self._mark(turn_id, "tts_request", ts, first_only=True)
            elif kind is EventType.TTS_COMPLETED:
                if event.reason == "cancelled":
                    active = self._active_turns.get(turn_id)
                    if active is not None and active["tts_stop_ts"] is None:
                        active["tts_stop_ts"] = ts
            elif kind is EventType.LLM_STARTED:
                self._mark(turn_id, "llm_start", ts, first_only=True)
            elif kind is EventType.LLM_TOKEN:
                self._mark(turn_id, "llm_first_token", ts, first_only=True)

        if kind is EventType.SPEECH_STARTED:
            self._speech_started_ts = ts
            self._partial_reported = False
        elif kind is EventType.TRANSCRIPT_PARTIAL:
            if self._speech_started_ts is not None and not self._partial_reported:
                self._utterance_ttfp.append(round(_ms(self._speech_started_ts, ts), 2))
                self._partial_reported = True
        elif kind is EventType.TRANSCRIPT_FINAL:
            if self._speech_started_ts is not None:
                self._utterance_ttf.append(round(_ms(self._speech_started_ts, ts), 2))
                self._speech_started_ts = None
                self._partial_reported = False

        if kind is EventType.USER_INTERRUPTED:
            self._counters["user_interrupts"] += 1
            interrupted = turn_id
            speech_ts = self._speech_started_ts
            active = self._active_turns.get(interrupted) if interrupted is not None else None
            if active is not None and speech_ts is not None:
                active["interrupt_detected_ms"] = round(_ms(speech_ts, ts), 2)
                active["interrupt_time"] = ts

        if kind is EventType.ERROR:
            self._counters["errors"] += 1
        elif kind is EventType.TOOL_CALL_STARTED:
            self._counters["tool_calls"] += 1
        elif kind is EventType.TOOL_CALL_FAILED:
            self._counters["tool_failures"] += 1

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict[str, Any]:
        latencies: dict[str, dict[str, float | None]] = {}
        for name, key in (
            ("ttft", "ttft_ms"),
            ("ttfa", "ttfa_ms"),
            ("e2e", "e2e_ms"),
            ("tts_request", "tts_request_ms"),
            ("interruption", "interrupt_detected_ms"),
            ("tts_cancellation", "tts_cancellation_ms"),
        ):
            latencies[name] = _aggregate([t.get(key) for t in self._completed_turns])

        return {
            "counters": {
                **dict(self._counters),
                "turns_in_flight": len(self._active_turns),
                "utterance_ttfp_n": len(self._utterance_ttfp),
                "utterance_ttf_n": len(self._utterance_ttf),
            },
            "latencies_ms": latencies,
            "transcript_ms": {
                "time_to_first_partial": _aggregate(self._utterance_ttfp),
                "time_to_final": _aggregate(self._utterance_ttf),
            },
        }
