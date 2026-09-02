from __future__ import annotations

import time
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    AUDIO_RECEIVED = "audio_received"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    LLM_STARTED = "llm_started"
    LLM_TOKEN = "llm_token"
    LLM_COMPLETED = "llm_completed"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
    TTS_STARTED = "tts_started"
    TTS_AUDIO = "tts_audio"
    TTS_COMPLETED = "tts_completed"
    USER_INTERRUPTED = "user_interrupted"
    AGENT_INTERRUPTED = "agent_interrupted"
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    RUNTIME_STATE_CHANGED = "runtime_state_changed"
    ERROR = "error"


class VoiceEvent(BaseModel):
    """Structural base shared by every event that flows through the runtime.

    Concrete event classes fix ``type`` and add payload fields. Every event is
    correlated by ``session_id`` and, when produced inside a turn, ``turn_id`` so
    consumers can reject stale events that outlive their owning turn.
    """

    model_config = ConfigDict(extra="forbid")

    type: EventType
    session_id: str
    conversation_id: str | None = None
    turn_id: int | None = None
    timestamp: float = Field(default_factory=time.time)
    latency_ms: float | None = None
    provider: str | None = None


class AudioReceived(VoiceEvent):
    type: EventType = EventType.AUDIO_RECEIVED
    sample_rate: int = 16_000
    num_samples: int = 0
    rms: float | None = None


class SpeechStarted(VoiceEvent):
    type: EventType = EventType.SPEECH_STARTED
    energy_db: float | None = None


class SpeechEnded(VoiceEvent):
    type: EventType = EventType.SPEECH_ENDED
    speech_duration_ms: float | None = None
    silence_duration_ms: float | None = None


class TranscriptPartial(VoiceEvent):
    type: EventType = EventType.TRANSCRIPT_PARTIAL
    text: str = ""
    is_final: bool = False
    words: list[dict[str, Any]] | None = None


class TranscriptFinal(VoiceEvent):
    type: EventType = EventType.TRANSCRIPT_FINAL
    text: str = ""
    is_final: bool = True
    words: list[dict[str, Any]] | None = None


class LLMStarted(VoiceEvent):
    type: EventType = EventType.LLM_STARTED
    request_id: str | None = None


class LLMToken(VoiceEvent):
    type: EventType = EventType.LLM_TOKEN
    text: str = ""
    index: int | None = None


class LLMCompleted(VoiceEvent):
    type: EventType = EventType.LLM_COMPLETED
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


class ToolCallStarted(VoiceEvent):
    type: EventType = EventType.TOOL_CALL_STARTED
    tool_name: str
    call_id: str | None = None
    arguments: str | None = None


class ToolCallCompleted(VoiceEvent):
    type: EventType = EventType.TOOL_CALL_COMPLETED
    tool_name: str
    call_id: str | None = None
    duration_ms: float | None = None
    result: str | dict[str, Any] | None = None


class ToolCallFailed(VoiceEvent):
    type: EventType = EventType.TOOL_CALL_FAILED
    tool_name: str
    call_id: str | None = None
    error: str = ""
    retryable: bool = False


class TTSStarted(VoiceEvent):
    type: EventType = EventType.TTS_STARTED
    text: str = ""
    request_id: str | None = None


class TTSAudio(VoiceEvent):
    type: EventType = EventType.TTS_AUDIO
    duration_ms: float | None = None
    audio_bytes: int = 0
    sequence: int | None = None


class TTSCompleted(VoiceEvent):
    type: EventType = EventType.TTS_COMPLETED
    reason: Literal["completed", "cancelled", "error"] = "completed"


class UserInterrupted(VoiceEvent):
    type: EventType = EventType.USER_INTERRUPTED
    interrupted_turn_id: int | None = None


class AgentInterrupted(VoiceEvent):
    type: EventType = EventType.AGENT_INTERRUPTED
    interrupted_turn_id: int | None = None


class TurnStarted(VoiceEvent):
    type: EventType = EventType.TURN_STARTED
    turn_type: Literal["user", "agent"] = "user"
    reason: str | None = None


class TurnCompleted(VoiceEvent):
    type: EventType = EventType.TURN_COMPLETED
    outcome: Literal["completed", "cancelled", "error"] = "completed"
    reason: str | None = None


class RuntimeStateChanged(VoiceEvent):
    type: EventType = EventType.RUNTIME_STATE_CHANGED
    from_state: str | None = None
    to_state: str


class ErrorEvent(VoiceEvent):
    type: EventType = EventType.ERROR
    error: str = ""
    stage: str | None = None
    fatal: bool = False
    exc_type: str | None = None


EVENT_CLASSES: dict[EventType, type[VoiceEvent]] = {
    cls.model_fields["type"].default: cls
    for cls in (
        AudioReceived,
        SpeechStarted,
        SpeechEnded,
        TranscriptPartial,
        TranscriptFinal,
        LLMStarted,
        LLMToken,
        LLMCompleted,
        ToolCallStarted,
        ToolCallCompleted,
        ToolCallFailed,
        TTSStarted,
        TTSAudio,
        TTSCompleted,
        UserInterrupted,
        AgentInterrupted,
        TurnStarted,
        TurnCompleted,
        RuntimeStateChanged,
        ErrorEvent,
    )
}


def event_from_dict(data: dict[str, Any]) -> VoiceEvent:
    event_type = EventType(data["type"])
    cls = EVENT_CLASSES[event_type]
    return cls.model_validate(data)
