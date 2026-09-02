from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.runtime import events as ev
from app.runtime.events import EventType, TranscriptFinal, event_from_dict


def test_all_event_types_are_registered() -> None:
    assert set(ev.EVENT_CLASSES) == set(EventType)
    for event_type, cls in ev.EVENT_CLASSES.items():
        assert cls.model_fields["type"].default is event_type


def test_event_defaults() -> None:
    event = TranscriptFinal(session_id="s1", text="hello")
    assert event.type is EventType.TRANSCRIPT_FINAL
    assert event.turn_id is None
    assert event.timestamp > 0


def test_transcript_final_json_round_trip() -> None:
    original = TranscriptFinal(
        session_id="s1",
        conversation_id="c1",
        turn_id=7,
        text="my payment failed",
        latency_ms=310.5,
        provider="deepgram",
    )
    payload = json.loads(original.model_dump_json())
    restored = event_from_dict(payload)
    assert isinstance(restored, TranscriptFinal)
    assert restored == original


def test_tool_event_round_trip_with_payload() -> None:
    event = ev.ToolCallStarted(
        session_id="s1",
        turn_id=3,
        tool_name="inspect_payment",
        call_id="call_1",
        arguments='{"payment_id": "pay_123"}',
    )
    restored = event_from_dict(json.loads(event.model_dump_json()))
    assert isinstance(restored, ev.ToolCallStarted)
    assert restored.tool_name == "inspect_payment"
    assert restored.call_id == "call_1"


def test_unknown_event_type_rejected() -> None:
    with pytest.raises((KeyError, ValueError)):
        event_from_dict({"type": "does_not_exist", "session_id": "s1"})


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        TranscriptFinal(session_id="s1", text="x", unexpected="nope")


def test_audio_event_carries_correlation_fields() -> None:
    event = ev.AudioReceived(session_id="s1", turn_id=1, sample_rate=16_000, num_samples=320)
    assert event.session_id == "s1"
    assert event.turn_id == 1
    assert event.type is EventType.AUDIO_RECEIVED


def test_error_event_defaults() -> None:
    event = ev.ErrorEvent(session_id="s1", error="boom")
    assert event.fatal is False
    assert event.stage is None
