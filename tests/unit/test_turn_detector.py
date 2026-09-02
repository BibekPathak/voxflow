from __future__ import annotations

import asyncio

import pytest

from app.runtime.events import SpeechEnded, SpeechStarted, TranscriptFinal, TranscriptPartial
from app.runtime.turn import TurnDetector, TurnDetectorParams


def _params(**overrides: object) -> TurnDetectorParams:
    base: dict[str, object] = dict(min_speech_ms=80, silence_ms=150, max_utterance_s=30, require_text=True)
    base.update(overrides)
    return TurnDetectorParams(**base)  # type: ignore[arg-type]


async def _wait_until(condition, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


async def test_submits_turn_after_silence_with_transcript() -> None:
    results: list[str] = []
    det = TurnDetector(session_id="s1", params=_params(silence_ms=120), on_turn_final=lambda t: results.append(t))

    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(TranscriptPartial(session_id="s1", text="my payment failed."))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=900))
    await _wait_until(lambda: len(results) == 1)

    assert results == ["my payment failed."]
    assert det.stats["turns_finalized"] == 1
    await det.close()


async def test_strong_endpoint_shortens_silence() -> None:
    results: list[str] = []
    det = TurnDetector(session_id="s1", params=_params(silence_ms=500), on_turn_final=lambda t: results.append(t))

    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(TranscriptFinal(session_id="s1", text="yes."))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=400))

    started = asyncio.get_running_loop().time()
    await _wait_until(lambda: len(results) == 1, timeout=1.5)
    elapsed = asyncio.get_running_loop().time() - started
    assert results == ["yes."]
    assert elapsed < 0.4
    await det.close()


async def test_short_speech_does_not_submit() -> None:
    results: list[str] = []
    det = TurnDetector(session_id="s1", params=_params(), on_turn_final=lambda t: results.append(t))

    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(TranscriptPartial(session_id="s1", text="hm"))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=50))
    await asyncio.sleep(0.35)

    assert results == []
    assert det.stats["short_segments"] == 1
    await det.close()


async def test_no_submit_without_transcript_when_required() -> None:
    results: list[str] = []
    det = TurnDetector(session_id="s1", params=_params(silence_ms=80), on_turn_final=lambda t: results.append(t))

    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=800))
    await asyncio.sleep(0.3)

    assert results == []
    await det.close()


async def test_submits_without_transcript_when_not_required() -> None:
    results: list[str] = []
    det = TurnDetector(
        session_id="s1",
        params=_params(silence_ms=80, require_text=False),
        on_turn_final=lambda t: results.append(t),
    )

    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=800))
    await _wait_until(lambda: len(results) == 1)
    assert results == [""]
    await det.close()


async def test_new_speech_cancels_pending_silence_timer() -> None:
    results: list[str] = []
    det = TurnDetector(session_id="s1", params=_params(silence_ms=150), on_turn_final=lambda t: results.append(t))

    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(TranscriptPartial(session_id="s1", text="I need to change my"))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=600))
    await asyncio.sleep(0.05)

    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(TranscriptPartial(session_id="s1", text="I need to change my billing address"))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=600))
    await _wait_until(lambda: len(results) == 1)

    assert results == ["I need to change my billing address"]
    assert det.stats["speech_segments"] == 2
    await det.close()


async def test_max_utterance_forces_submit() -> None:
    results: list[str] = []
    det = TurnDetector(
        session_id="s1",
        params=_params(max_utterance_s=1, require_text=False),
        on_turn_final=lambda t: results.append(t),
    )

    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=0))
    await det.handle(SpeechStarted(session_id="s1"))

    await _wait_until(lambda: len(results) == 1, timeout=2.0)
    assert results == [""]
    assert det.stats["forced_submits"] == 1
    await det.close()


async def test_duplicate_and_late_events_are_ignored() -> None:
    results: list[str] = []
    det = TurnDetector(session_id="s1", params=_params(silence_ms=80), on_turn_final=lambda t: results.append(t))

    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(TranscriptPartial(session_id="s1", text="go ahead"))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=900))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=900))
    await _wait_until(lambda: len(results) == 1)

    assert len(results) == 1
    assert det.stats["speech_segments"] == 1
    await det.close()


async def test_close_cancels_pending_timers() -> None:
    results: list[str] = []
    det = TurnDetector(session_id="s1", params=_params(silence_ms=500), on_turn_final=lambda t: results.append(t))

    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(TranscriptPartial(session_id="s1", text="almost done"))
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=800))
    await det.close()
    await asyncio.sleep(0.6)

    assert results == []


async def test_invalid_speech_end_before_start_is_ignored() -> None:
    det = TurnDetector(session_id="s1", params=_params(), on_turn_final=None)
    await det.handle(SpeechEnded(session_id="s1", speech_duration_ms=900))
    assert det.stats["speech_segments"] == 0
    await det.close()


@pytest.mark.parametrize("text", ["no", "hm", "yes, but", "fine"])
async def test_partial_updates_detector_text(text: str) -> None:
    det = TurnDetector(session_id="s1", params=_params(), on_turn_final=None)
    await det.handle(SpeechStarted(session_id="s1"))
    await det.handle(TranscriptPartial(session_id="s1", text=text))
    assert det._has_transcript is True
    await det.close()
