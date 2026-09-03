from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol

from app.audio.gateway import AudioGateway
from app.config import Settings
from app.memory.conversation import ConversationStore, HistoryEntry
from app.observability.logging import get_logger, log_event
from app.observability.metrics import MetricsCollector
from app.providers.factory import ProviderSet, build_providers
from app.providers.types import AudioData, ToolSpec, Transcript
from app.runtime.cancellation import CancellationScope, TurnCancelled
from app.runtime.errors import StateViolationError
from app.runtime.event_bus import EventBus, EventSubscription
from app.runtime.events import (
    AudioReceived,
    ErrorEvent,
    RuntimeStateChanged,
    SpeechEnded,
    SpeechStarted,
    TranscriptFinal,
    TranscriptPartial,
    TTSAudio,
    TurnCompleted,
    TurnStarted,
    UserInterrupted,
    VoiceEvent,
)
from app.runtime.pipeline import StreamingTurnPipeline
from app.runtime.state_machine import RuntimeState, StateMachine
from app.runtime.turn import TurnDetector, TurnDetectorParams
from app.tools import load_builtin_tools
from app.tools.registry import ToolRegistry

_logger = get_logger("voxflow.runtime")


class _SttEnd:
    pass


_STT_END = _SttEnd()


class TurnContext:
    """Everything a pipeline needs to execute one user turn."""

    __slots__ = ("session_id", "conversation_id", "turn_id", "text", "started_at", "response_text")

    def __init__(self, *, session_id: str, conversation_id: str, turn_id: int, text: str) -> None:
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.turn_id = turn_id
        self.text = text
        self.started_at = time.monotonic()
        self.response_text = ""


class TurnPipeline(Protocol):
    """Streams one user turn through LLM/tools/TTS.

    The runtime owns state, cancellation and eventing; a pipeline implementation
    only drives provider calls. This keeps providers swappable without touching
    the session lifecycle.
    """

    async def handle(self, runtime: SessionRuntime, context: TurnContext) -> None: ...


class AudioOutbound(Protocol):
    """Full-duplex audio socket interface used to push agent audio to the browser."""

    async def send_text(self, data: str) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...


class SessionRuntime:
    """The per-session async engine.

    Responsibilities:

    * own the session event bus and its state machine;
    * ingest inbound PCM through the audio gateway, forwarding it simultaneously
      to the energy VAD (speech events + endpointing) and the streaming STT
      provider (partial/final transcripts);
    * drive user-turn lifecycle (TurnStarted -> LLM/TTS pipeline -> TurnCompleted)
      inside a per-turn cancellation scope, so interruptions cancel cleanly;
    * stream agent audio back to the browser with guard rails that prevent stale
      audio from surviving an interruption;
    * expose the state snapshot the API/dashboard read.

    A session has one audio owner at a time. When the owner disconnects the
    runtime cancels in-flight work and transitions to CLOSED (terminal), which
    prevents TTS from being started into a dead session.
    """

    def __init__(
        self,
        session_id: str,
        conversation_id: str,
        settings: Settings,
        providers: ProviderSet | None = None,
        tools: ToolRegistry | None = None,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.settings = settings
        self.providers = providers if providers is not None else build_providers(settings)
        self.tools = tools if tools is not None else load_builtin_tools()
        self._store = conversation_store

        self.bus = EventBus(name=f"session:{session_id}", queue_maxsize=settings.audio_queue_maxsize)
        self.states = StateMachine(on_change=self._on_state_changed)

        self.gateway = AudioGateway(
            sample_rate=settings.sample_rate,
            vad_speech_start_rms=settings.vad_speech_start_rms,
            vad_speech_end_rms=settings.vad_speech_end_rms,
            vad_start_confirm_ms=settings.vad_start_confirm_ms,
            vad_end_confirm_ms=settings.vad_end_confirm_ms,
            vad_frame_ms=settings.vad_frame_ms,
        )

        self.detector = TurnDetector(
            session_id=session_id,
            params=TurnDetectorParams(
                min_speech_ms=settings.turn_min_speech_ms,
                silence_ms=settings.turn_silence_ms,
                max_utterance_s=settings.turn_max_utterance_s,
                require_text=True,
            ),
            on_turn_final=self._finalize_user_turn,
        )
        self._detector_sub: EventSubscription = self.bus.subscribe(
            self.detector.handle,
            session_id=session_id,
            event_types=self.detector.handles(),
        )

        self.metrics = MetricsCollector()
        self._metrics_sub: EventSubscription = self.bus.subscribe(
            self.metrics.on_event,
            session_id=session_id,
        )

        self.pipeline: TurnPipeline = StreamingTurnPipeline()

        self.created_at = time.time()
        self._history: list[HistoryEntry] = []
        self._audio_owner: str | None = None
        self._outbound: AudioOutbound | None = None
        self._closed = False
        self.turn_count = 0
        self.current_turn_id: int | None = None
        self._active_scope: CancellationScope | None = None
        self._interrupted_active = False
        self._cancel_tasks: set[asyncio.Task[Any]] = set()
        self._last_error: dict[str, Any] | None = None
        self._audio_seq = 0

        self._stt_queue: asyncio.Queue[bytes | _SttEnd] | None = None
        self._stt_task: asyncio.Task[None] | None = None
        self._last_speech_end_ts: float | None = None

    @property
    def state(self) -> RuntimeState:
        return self.states.state

    @property
    def audio_connected(self) -> bool:
        return self._audio_owner is not None

    @property
    def active_scope(self) -> CancellationScope | None:
        return self._active_scope

    @property
    def history(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._history)

    def tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=definition.name, description=definition.description, parameters=definition.parameters)
            for definition in self.tools.definitions()
        ]

    def publish(self, event: VoiceEvent) -> None:
        self.bus.publish(event)
        log_event(event)

    def _on_state_changed(self, from_state: str, to_state: str, reason: str | None) -> None:
        self.publish(
            RuntimeStateChanged(
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
            )
        )

    # ------------------------------------------------------------------ audio
    def attach(self, owner: str, outbound: AudioOutbound | None = None) -> bool:
        if self._closed:
            return False
        if self._audio_owner is not None and self._audio_owner != owner:
            return False
        if self._audio_owner is None:
            try:
                self.states.transition(RuntimeState.LISTENING, reason="audio_connected")
            except StateViolationError:
                if self.states.state is not RuntimeState.LISTENING:
                    return False
            self._start_stt()
        self._audio_owner = owner
        self._outbound = outbound
        return True

    async def detach(self, owner: str) -> None:
        if self._audio_owner != owner:
            return
        self._audio_owner = None
        self._outbound = None
        self._interrupted_active = False
        force_end = self.gateway.vad.force_end()
        if force_end is not None:
            await self._handle_vad_decision(force_end)
        await self.close("audio_disconnected")

    async def close(self, reason: str = "session_closed") -> None:
        """Terminate the session: cancel in-flight work and transition to CLOSED.

        The event bus is intentionally left open so observers (e.g. the events
        WebSocket) receive the terminal CLOSED state change. Call :meth:`dispose`
        to fully tear the session down.
        """
        if self._closed:
            return
        self._closed = True
        self._interrupted_active = False
        await self._stop_stt()
        await self._cancel_active_scope()
        await self.detector.close()
        self._detector_sub.close()
        self._metrics_sub.close()
        try:
            self.states.transition(RuntimeState.CLOSED, reason=reason)
        except StateViolationError:
            pass

    async def dispose(self, reason: str = "manager_shutdown") -> None:
        await self.close(reason)
        await self.bus.close()
        await self.providers.close()

    async def ingest_audio(self, data: bytes, *, sequence: int | None = None) -> None:
        if self._closed:
            return
        self._push_stt_audio(data)
        result = self.gateway.ingest_pcm(data, sequence=sequence)
        if result.num_samples == 0:
            return
        self.publish(
            AudioReceived(
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                turn_id=self.current_turn_id,
                sample_rate=self.gateway.sample_rate,
                num_samples=result.num_samples,
                rms=result.rms,
            )
        )
        for decision in result.decisions:
            await self._handle_vad_decision(decision)

    # -------------------------------------------------------------------- STT
    def _start_stt(self) -> None:
        if self._stt_task is not None:
            return
        self._stt_queue = asyncio.Queue()
        self._stt_task = asyncio.create_task(self._stt_run(), name=f"stt-{self.session_id[:8]}")

    async def _stop_stt(self) -> None:
        task = self._stt_task
        self._stt_task = None
        self._stt_queue = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _push_stt_audio(self, data: bytes) -> None:
        queue = self._stt_queue
        if queue is None or not data:
            return
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            pass

    async def _stt_run(self) -> None:
        try:

            async def audio_iter() -> AsyncIterator[bytes]:
                while True:
                    item = await self._stt_queue.get()  # type: ignore[union-attr]
                    if isinstance(item, _SttEnd):
                        return
                    yield item

            async for transcript in self.providers.stt.transcribe_stream(
                audio_iter(),
                sample_rate=self.gateway.sample_rate,
                interim_results=True,
            ):
                await self._handle_stt_transcript(transcript)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                _logger.warning("stt stream ended unexpectedly", error=str(exc), session_id=self.session_id)
                self.publish(
                    ErrorEvent(
                        session_id=self.session_id,
                        conversation_id=self.conversation_id,
                        error=str(exc),
                        stage="stt",
                        exc_type=type(exc).__name__,
                    )
                )

    async def _handle_stt_transcript(self, transcript: Transcript) -> None:
        if not transcript.text.strip():
            return
        cls = TranscriptFinal if transcript.is_final else TranscriptPartial
        self.publish(
            cls(
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                turn_id=self.current_turn_id,
                text=transcript.text,
                words=transcript.words,
                provider=self.settings.provider_stt,
            )
        )

    # ------------------------------------------------------------------ VAD
    async def _handle_vad_decision(self, decision: Any) -> None:
        kind = decision.kind
        if kind == "speech_start":
            await self._on_speech_start(decision)
        elif kind == "speech_end":
            await self._on_speech_end(decision)

    async def _on_speech_start(self, decision: Any) -> None:
        self.publish(
            SpeechStarted(
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                turn_id=self.current_turn_id,
                energy_db=decision.energy_db,
            )
        )
        if self.states.state in (
            RuntimeState.SPEAKING,
            RuntimeState.PROCESSING,
            RuntimeState.THINKING,
        ):
            await self._begin_interruption()

    async def _begin_interruption(self) -> None:
        interrupted_turn = self.current_turn_id
        try:
            self.states.transition(RuntimeState.INTERRUPTED, reason="barge_in")
        except StateViolationError:
            return
        self._interrupted_active = True
        self.publish(
            UserInterrupted(
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                turn_id=interrupted_turn,
                interrupted_turn_id=interrupted_turn,
                latency_ms=self._interruption_latency_ms(),
            )
        )
        await self._cancel_active_scope()

    def _interruption_latency_ms(self) -> float | None:
        return None

    async def _on_speech_end(self, decision: Any) -> None:
        self._last_speech_end_ts = time.time()
        self.publish(
            SpeechEnded(
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                turn_id=self.current_turn_id,
                speech_duration_ms=decision.speech_duration_ms,
            )
        )
        if self._interrupted_active:
            self._interrupted_active = False
            try:
                self.states.transition(RuntimeState.LISTENING, reason="interrupt_speech_ended")
            except StateViolationError:
                pass

    # ------------------------------------------------------------ agent audio
    def is_audio_emittable(self, turn_id: int) -> bool:
        if self._closed or self.current_turn_id != turn_id:
            return False
        return self.states.state in (RuntimeState.PROCESSING, RuntimeState.SPEAKING)

    def _transition_speaking(self) -> bool:
        try:
            if self.states.state is RuntimeState.PROCESSING:
                self.states.transition(RuntimeState.SPEAKING, reason="tts_first_audio")
            return True
        except StateViolationError:
            return False

    async def begin_agent_audio(self, turn_id: int) -> None:
        self._transition_speaking()
        if self._outbound is not None:
            await self._outbound_safe_text(json.dumps({"type": "agent_audio.start", "turn_id": turn_id}))

    async def end_agent_audio(self, turn_id: int, *, reason: str, started: bool) -> None:
        if self._outbound is not None and started:
            await self._outbound_safe_text(
                json.dumps({"type": "agent_audio.end", "turn_id": turn_id, "reason": reason})
            )

    async def emit_agent_audio(self, turn_id: int, audio: AudioData) -> bool:
        if not self.is_audio_emittable(turn_id):
            return False
        self._audio_seq += 1
        self.publish(
            TTSAudio(
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                turn_id=turn_id,
                duration_ms=audio.duration_ms,
                audio_bytes=len(audio.pcm),
                sequence=self._audio_seq,
                provider=self.settings.provider_tts,
            )
        )
        if self._outbound is not None and audio.pcm:
            await self._outbound_safe_bytes(audio.pcm)
        return True

    async def _outbound_safe_text(self, data: str) -> None:
        outbound = self._outbound
        if outbound is None or self._closed:
            return
        try:
            await outbound.send_text(data)
        except Exception as exc:
            _logger.warning("outbound text failed", error=str(exc), session_id=self.session_id)

    async def _outbound_safe_bytes(self, data: bytes) -> None:
        outbound = self._outbound
        if outbound is None or self._closed:
            return
        try:
            await outbound.send_bytes(data)
        except Exception as exc:
            _logger.warning("outbound audio failed", error=str(exc), session_id=self.session_id)

    # ------------------------------------------------------------------ turn
    async def _finalize_user_turn(self, text: str) -> None:
        if self._closed or self.states.state is not RuntimeState.LISTENING:
            return
        self.turn_count += 1
        turn_id = self.turn_count
        self.current_turn_id = turn_id
        self.states.transition(RuntimeState.PROCESSING, reason=f"user_turn:{turn_id}")

        scope = CancellationScope(turn_id=turn_id, name=f"turn-{turn_id}")
        self._active_scope = scope
        context = TurnContext(
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            turn_id=turn_id,
            text=text,
        )
        self.publish(
            TurnStarted(
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                turn_id=turn_id,
                turn_type="user",
                reason="stt_final",
            )
        )
        self.metrics.register_turn(turn_id, speech_end_ts=self._last_speech_end_ts)
        try:
            await scope.run(self._run_pipeline(context))
            await scope.close()
            await self._commit_turn(context)
            self._publish_turn_completed(turn_id, "completed")
        except TurnCancelled:
            self._publish_turn_completed(turn_id, "cancelled", reason="interrupted")
        except asyncio.CancelledError:
            self._publish_turn_completed(turn_id, "cancelled", reason="session_closed")
            raise
        except Exception as exc:
            self._record_error(turn_id, exc, stage="pipeline")
            self._publish_turn_completed(turn_id, "error", reason=str(exc))
        finally:
            self._active_scope = None
            self.current_turn_id = None
            try:
                if self.states.state in (RuntimeState.PROCESSING, RuntimeState.SPEAKING):
                    self.states.transition(RuntimeState.LISTENING, reason=f"turn:{turn_id}_complete")
            except StateViolationError:
                pass

    async def _run_pipeline(self, context: TurnContext) -> None:
        await self.pipeline.handle(self, context)

    async def _commit_turn(self, context: TurnContext) -> None:
        user_text = context.text.strip()
        if not user_text and not context.response_text:
            return
        self._history.append(HistoryEntry(role="user", content=user_text or "", turn_ordinal=context.turn_id))
        if context.response_text.strip():
            self._history.append(
                HistoryEntry(role="assistant", content=context.response_text.strip(), turn_ordinal=context.turn_id)
            )
        if self._store is None or self.settings.persist_conversations is False:
            return
        try:
            await self._store.add_message(
                conversation_id=self.conversation_id,
                session_id=self.session_id,
                turn_ordinal=context.turn_id,
                role="user",
                content=user_text,
            )
            if context.response_text.strip():
                await self._store.add_message(
                    conversation_id=self.conversation_id,
                    session_id=self.session_id,
                    turn_ordinal=context.turn_id,
                    role="assistant",
                    content=context.response_text.strip(),
                )
        except Exception as exc:
            _logger.warning("failed to persist turn", error=str(exc), session_id=self.session_id)

    def _publish_turn_completed(self, turn_id: int, outcome: str, *, reason: str | None = None) -> None:
        self.publish(
            TurnCompleted(
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                turn_id=turn_id,
                outcome=outcome,
                reason=reason,
            )
        )

    def _record_error(self, turn_id: int, exc: Exception, *, stage: str) -> None:
        self._last_error = {
            "stage": stage,
            "error": str(exc),
            "type": type(exc).__name__,
            "turn_id": turn_id,
        }
        self.publish(
            ErrorEvent(
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                turn_id=turn_id,
                error=str(exc),
                stage=stage,
                exc_type=type(exc).__name__,
            )
        )

    async def _cancel_active_scope(self) -> None:
        scope = self._active_scope
        if scope is None:
            return
        task = asyncio.create_task(scope.cancel(), name=f"cancel-turn-{scope.turn_id}")
        self._cancel_tasks.add(task)
        task.add_done_callback(self._cancel_tasks.discard)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "state": self.states.state.value,
            "audio_connected": self.audio_connected,
            "created_at": self.created_at,
            "turn_count": self.turn_count,
            "current_turn_id": self.current_turn_id,
            "vad": {
                "speech_active": self.gateway.vad.speech_active,
                "total_speech_ms": self.gateway.vad.total_speech_ms,
            },
            "turns": self.detector.stats,
            "last_error": self._last_error,
        }
