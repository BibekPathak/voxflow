from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from app.memory.context import build_context_messages
from app.providers.types import LLMMessage, ToolSpec
from app.runtime.events import (
    LLMCompleted,
    LLMStarted,
    LLMToken,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    TTSCompleted,
    TTSStarted,
)
from app.runtime.text_chunker import SentenceChunker

if TYPE_CHECKING:
    from app.runtime.orchestrator import SessionRuntime, TurnContext

SYSTEM_PROMPT = (
    "You are VoxFlow, a friendly voice support agent for a payments company. "
    "The user is speaking to you. Respond out loud, so keep answers short, "
    "conversational and natural. Do not use markdown or lists. Use the provided "
    "tools to look up customer and payment details before answering."
)

MAX_TOOL_ROUNDS = 4


async def _queue_texts(queue: asyncio.Queue[str | None]) -> AsyncIterator[str]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


class StreamingTurnPipeline:
    """Drives one user turn through the streaming provider stack.

    A conversation pass streams LLM tokens into the sentence chunker, which
    feeds speakable phrases to a concurrent streaming TTS worker so audio
    synthesis overlaps remaining generation. If the LLM instead requests tools,
    the tools execute (with timeout/retry) and the results feed another LLM
    pass, repeated until the model produces its spoken answer. Everything runs
    under the turn's cancellation scope; stale audio can never reach the browser
    after an interruption.
    """

    def __init__(self) -> None:
        pass

    async def handle(self, runtime: SessionRuntime, context: TurnContext) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=64)
        scope = runtime.active_scope
        if scope is not None:
            tts_task = scope.spawn(self._run_tts(runtime, context, queue), name=f"tts-{context.turn_id}")
        else:
            tts_task = asyncio.create_task(self._run_tts(runtime, context, queue))

        try:
            await self._run(runtime, context, queue)
            await tts_task
        except BaseException:
            if not tts_task.done():
                tts_task.cancel()
            raise
        finally:
            if not tts_task.done():
                tts_task.cancel()
            await asyncio.gather(tts_task, return_exceptions=True)

    async def _run(self, runtime: SessionRuntime, context: TurnContext, queue: asyncio.Queue[str | None]) -> None:
        messages = build_context_messages(
            runtime.history,
            system_prompt=SYSTEM_PROMPT,
            user_text=context.text,
            max_turns=runtime.settings.context_max_turns,
            max_chars=runtime.settings.context_max_chars,
        )
        runtime.publish(
            LLMStarted(
                session_id=context.session_id,
                conversation_id=context.conversation_id,
                turn_id=context.turn_id,
                provider=runtime.settings.provider_llm,
            )
        )
        try:
            for _round in range(MAX_TOOL_ROUNDS):
                tools: list[ToolSpec] = runtime.tool_specs() if _round == 0 else []
                result = await self._llm_pass(runtime, context, queue, messages, tools)

                if result["tool_calls"] and result["finish_reason"] == "tool_calls":
                    await self._execute_tool_calls(runtime, context, messages, result)
                    continue

                for piece in result["chunker"].flush():
                    await queue.put(piece)
                context.response_text = result["text"]
                runtime.publish(
                    LLMCompleted(
                        session_id=context.session_id,
                        conversation_id=context.conversation_id,
                        turn_id=context.turn_id,
                        finish_reason=result["finish_reason"],
                    )
                )
                return
            raise RuntimeError("LLM exceeded the maximum number of tool-call rounds")
        finally:
            await queue.put(None)

    async def _llm_pass(
        self,
        runtime: SessionRuntime,
        context: TurnContext,
        queue: asyncio.Queue[str | None],
        messages: list[LLMMessage],
        tools: list[ToolSpec],
    ) -> dict[str, Any]:
        chunker = SentenceChunker()
        accumulators: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        text_parts: list[str] = []
        token_index = 0
        async for chunk in runtime.providers.llm.stream_chat(messages, tools=tools or None):
            if chunk.text:
                text_parts.append(chunk.text)
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
            if chunk.tool_call is not None:
                tc = chunk.tool_call
                acc = accumulators.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    acc["id"] += tc.id
                if tc.name:
                    acc["name"] += tc.name
                if tc.arguments:
                    acc["arguments"] += tc.arguments
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

        tool_calls = [
            {"id": acc["id"], "name": acc["name"], "arguments": acc["arguments"]}
            for acc in accumulators.values()
            if acc["name"]
        ]
        return {
            "chunker": chunker,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
            "text": "".join(text_parts),
        }

    async def _execute_tool_calls(
        self,
        runtime: SessionRuntime,
        context: TurnContext,
        messages: list[LLMMessage],
        result: dict[str, Any],
    ) -> None:
        calls = result["tool_calls"]
        formatted = [
            {
                "id": call["id"] or f"call_{index}",
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"] or "{}"},
            }
            for index, call in enumerate(calls)
        ]
        messages.append(LLMMessage(role="assistant", content=result["text"], tool_calls=formatted))
        for call in formatted:
            name = call["function"]["name"]
            call_id = call["id"]
            args = call["function"]["arguments"]
            runtime.publish(
                ToolCallStarted(
                    session_id=context.session_id,
                    conversation_id=context.conversation_id,
                    turn_id=context.turn_id,
                    tool_name=name,
                    call_id=call_id,
                    arguments=args,
                )
            )
            outcome = await runtime.tools.execute(
                name,
                args,
                default_timeout_s=runtime.settings.tool_timeout_s,
                default_retries=runtime.settings.tool_max_retries,
            )
            if outcome.ok:
                runtime.publish(
                    ToolCallCompleted(
                        session_id=context.session_id,
                        conversation_id=context.conversation_id,
                        turn_id=context.turn_id,
                        tool_name=name,
                        call_id=call_id,
                        duration_ms=outcome.duration_ms,
                        result=outcome.value,
                    )
                )
            else:
                runtime.publish(
                    ToolCallFailed(
                        session_id=context.session_id,
                        conversation_id=context.conversation_id,
                        turn_id=context.turn_id,
                        tool_name=name,
                        call_id=call_id,
                        error=outcome.error or "",
                    )
                )
            messages.append(LLMMessage(role="tool", tool_call_id=call_id, content=outcome.content))

    async def _run_tts(self, runtime: SessionRuntime, context: TurnContext, queue: asyncio.Queue[str | None]) -> None:
        tts = runtime.providers.tts
        audio_started = False
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
