from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.runtime.events import EventType, SpeechEnded, VoiceEvent

TurnFinalCallback = Callable[[str, float | None], Awaitable[None]]


@dataclass(slots=True)
class TurnDetectorParams:
    min_speech_ms: int = 400
    silence_ms: int = 750
    max_utterance_s: int = 30
    require_text: bool = True
    endpoint_punctuation: frozenset[str] = frozenset({".", "?", "!"})


class TurnDetector:
    """Decides when the user has finished speaking (endpointing).

    Endpointing is deliberately conservative. It never fires on raw VAD silence
    alone; it combines:

    * VAD speech-end (a confirmed speech segment),
    * a configurable trailing-silence window (the user has stopped talking),
    * a minimum speech duration (so coughs/backchannels do not end a turn),
    * transcript state when available (a strong endpoint like a final STT
      transcript or ending punctuation can shorten the required silence),
    * a maximum-utterance cap so a talkative user still gets processed.

    Mid-sentence pauses are tolerated: a pause inside an utterance does not
    submit a fragment, because the speech-end handler only runs after VAD has
    confirmed real speech, and submission waits out the trailing silence.

    The detector is event-driven: it subscribes to the session event bus, so all
    decisions are made in event order on the bus worker.
    """

    def __init__(
        self,
        *,
        session_id: str,
        params: TurnDetectorParams | None = None,
        on_turn_final: TurnFinalCallback | None = None,
    ) -> None:
        self.session_id = session_id
        self.params = params or TurnDetectorParams()
        self._on_turn_final = on_turn_final

        self._utterance_seq = 0
        self._speech_active = False
        self._has_transcript = False
        self._text = ""
        self._silence_task: asyncio.Task[Any] | None = None
        self._max_task: asyncio.Task[Any] | None = None
        self._closed = False

        self.stats = {
            "speech_segments": 0,
            "turns_finalized": 0,
            "forced_submits": 0,
            "short_segments": 0,
        }

    def handles(self) -> frozenset[EventType]:
        return frozenset(
            {
                EventType.SPEECH_STARTED,
                EventType.SPEECH_ENDED,
                EventType.TRANSCRIPT_PARTIAL,
                EventType.TRANSCRIPT_FINAL,
            }
        )

    async def handle(self, event: VoiceEvent) -> None:
        """Invoked in event order by the session event-bus worker."""
        if event.type is EventType.SPEECH_STARTED:
            await self._on_speech_started()
        elif event.type is EventType.SPEECH_ENDED:
            await self._on_speech_ended(event)
        elif event.type in (EventType.TRANSCRIPT_PARTIAL, EventType.TRANSCRIPT_FINAL):
            self._text = str(getattr(event, "text", ""))
            if self._text.strip():
                self._has_transcript = True

    async def close(self) -> None:
        self._closed = True
        for task in (self._silence_task, self._max_task):
            if task is not None and not task.done():
                task.cancel()
        if self._silence_task:
            await asyncio.gather(self._silence_task, return_exceptions=True)
        if self._max_task:
            await asyncio.gather(self._max_task, return_exceptions=True)

    async def _cancel_timers(self) -> None:
        current = asyncio.current_task()
        candidates = [
            task
            for task in (self._silence_task, self._max_task)
            if task is not None and task is not current and not task.done()
        ]
        for task in candidates:
            task.cancel()
        if candidates:
            await asyncio.gather(*candidates, return_exceptions=True)
        self._silence_task = None
        self._max_task = None

    async def _on_speech_started(self) -> None:
        await self._cancel_timers()
        if self._speech_active:
            return
        self._speech_active = True
        self._utterance_seq += 1
        self._text = ""
        self._has_transcript = False
        seq = self._utterance_seq
        if self.params.max_utterance_s > 0:
            self._max_task = asyncio.create_task(self._watch_max_utterance(seq), name=f"vox-max-{seq}")

    async def _on_speech_ended(self, event: SpeechEnded) -> None:
        if not self._speech_active:
            return
        self._speech_active = False
        self.stats["speech_segments"] += 1
        if self._max_task is not None and not self._max_task.done():
            self._max_task.cancel()
        self._max_task = None

        duration_ms = event.speech_duration_ms
        if duration_ms is not None and duration_ms < self.params.min_speech_ms:
            self.stats["short_segments"] += 1
            return

        seq = self._utterance_seq
        if self.params.require_text:
            wait_s = self._silence_when_text_pending()
        else:
            wait_s = self.params.silence_ms / 1000.0
        self._silence_task = asyncio.create_task(self._watch_silence(seq, wait_s), name=f"vox-silence-{seq}")

    def _silence_when_text_pending(self) -> float:
        """Shorter trailing-silence window when STT already produced a final
        transcript with a strong endpoint; otherwise the full configured window."""
        if self._has_transcript and self._text.rstrip().endswith(tuple(self.params.endpoint_punctuation)):
            return max(0.05, self.params.silence_ms / 1000.0 / 3)
        return self.params.silence_ms / 1000.0

    async def _watch_silence(self, seq: int, wait_s: float) -> None:
        try:
            await asyncio.sleep(wait_s)
        except asyncio.CancelledError:
            return
        if self._closed or seq != self._utterance_seq or self._speech_active:
            return
        if self.params.require_text and not self._has_transcript:
            return
        await self._finalize(seq)

    async def _watch_max_utterance(self, seq: int) -> None:
        try:
            await asyncio.sleep(self.params.max_utterance_s)
        except asyncio.CancelledError:
            return
        if self._closed or seq != self._utterance_seq or not self._speech_active:
            return
        if self.params.require_text and not self._has_transcript:
            self._max_task = asyncio.create_task(self._watch_max_utterance(seq), name=f"vox-max-{seq}")
            return
        self.stats["forced_submits"] += 1
        await self._finalize(seq)

    async def _finalize(self, seq: int) -> None:
        await self._cancel_timers()
        if seq != self._utterance_seq or self._closed:
            return
        self._utterance_seq += 1
        self._speech_active = False
        self._text = self._text.strip()
        self.stats["turns_finalized"] += 1
        if self._on_turn_final is not None:
            await self._on_turn_final(self._text)
