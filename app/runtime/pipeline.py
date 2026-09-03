from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from app.providers.types import LLMMessage
from app.runtime.events import LLMCompleted, LLMStarted, LLMToken, TTSCompleted, TTSStarted
from app.runtime.text_chunker import SentenceChunker

if TYPE_CHECKING:
    from app.runtime.orchestrator import SessionRuntime, TurnContext

SYSTEM_PROMPT = (
    "You are VoxFlow, a friendly voice support agent for a payments company. "
    "The user is speaking to you. Respond out loud, so keep answers short, "
    "conversational and natural. Do not use markdown or lists."
)


def _build_messages(context: TurnContext) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=context.text),
    ]


async def _queue_texts(queue: asyncio.Queue[str | None]) -> AsyncIterator[str]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


class StreamingTurnPipeline:
    """Drives one user turn through the streaming provider stack.

    LLM tokens stream in; the sentence chunker groups them into speakable
    phrases; those phrases are handed to the streaming TTS provider over a queue
    so audio synthesis overlaps remaining LLM generation (the TTS worker runs
    concurrently). Audio frames are forwarded to the browser immediately.

    Every unit of work runs under the turn's cancellation scope, so a barge-in
    (or disconnect) cancels both the LLM consumer and the TTS worker together.
    The TTS worker guards each frame with a turn-id / state check so stale audio
    can never reach the browser after cancellation.
    """

    def __init__(self) -> None:
        pass

    async def handle(self, runtime: SessionRuntime, context: TurnContext) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=32)
        scope = runtime.active_scope
        if scope is not None:
            tts_task = scope.spawn(self._run_tts(runtime, context, queue), name=f"tts-{context.turn_id}")
        else:
            tts_task = asyncio.create_task(self._run_tts(runtime, context, queue))

        try:
            await self._drive_llm(runtime, context, queue)
            await tts_task
        except BaseException:
            if not tts_task.done():
                tts_task.cancel()
            raise
        finally:
            if not tts_task.done():
                tts_task.cancel()
            await asyncio.gather(tts_task, return_exceptions=True)

    async def _drive_llm(self, runtime: SessionRuntime, context: TurnContext, queue: asyncio.Queue[str | None]) -> None:
        llm = runtime.providers.llm
        chunker = SentenceChunker()
        runtime.publish(
            LLMStarted(
                session_id=context.session_id,
                conversation_id=context.conversation_id,
                turn_id=context.turn_id,
                provider=runtime.settings.provider_llm,
            )
        )
        finish_reason: str | None = None
        token_index = 0
        try:
            async for chunk in llm.stream_chat(_build_messages(context), tools=None):
                if chunk.text:
                    runtime.publish(
                        LLMToken(
                            session_id=context.session_id,
                            conversation_id=context.conversation_id,
                            turn_id=context.turn_id,
                            text=chunk.text,
                            index=token_index,
                        )
                    )
                    token_index += 1
                    for piece in chunker.push(chunk.text):
                        await queue.put(piece)
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
        finally:
            for piece in chunker.flush():
                await queue.put(piece)
            await queue.put(None)
        runtime.publish(
            LLMCompleted(
                session_id=context.session_id,
                conversation_id=context.conversation_id,
                turn_id=context.turn_id,
                finish_reason=finish_reason,
            )
        )

    async def _run_tts(self, runtime: SessionRuntime, context: TurnContext, queue: asyncio.Queue[str | None]) -> None:
        tts = runtime.providers.tts
        audio_started = False
        end_sent = False
        runtime.publish(
            TTSStarted(
                session_id=context.session_id,
                conversation_id=context.conversation_id,
                turn_id=context.turn_id,
                provider=runtime.settings.provider_tts,
            )
        )
        try:
            async for audio in tts.synthesize_stream(_queue_texts(queue), sample_rate=runtime.gateway.sample_rate):
                if not audio.pcm:
                    continue
                if not runtime.is_audio_emittable(context.turn_id):
                    break
                if not audio_started:
                    await runtime.begin_agent_audio(context.turn_id)
                    audio_started = True
                await runtime.emit_agent_audio(context.turn_id, audio)
            await runtime.end_agent_audio(context.turn_id, reason="completed", started=audio_started)
            end_sent = True
            runtime.publish(
                TTSCompleted(
                    session_id=context.session_id,
                    conversation_id=context.conversation_id,
                    turn_id=context.turn_id,
                    reason="completed",
                )
            )
        except asyncio.CancelledError:
            await runtime.end_agent_audio(context.turn_id, reason="cancelled", started=audio_started)
            runtime.publish(
                TTSCompleted(
                    session_id=context.session_id,
                    conversation_id=context.conversation_id,
                    turn_id=context.turn_id,
                    reason="cancelled",
                )
            )
            raise
        except Exception:
            if not end_sent:
                await runtime.end_agent_audio(context.turn_id, reason="error", started=audio_started)
            runtime.publish(
                TTSCompleted(
                    session_id=context.session_id,
                    conversation_id=context.conversation_id,
                    turn_id=context.turn_id,
                    reason="error",
                )
            )
            raise
