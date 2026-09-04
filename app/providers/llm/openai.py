from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from openai import AsyncOpenAI

from app.providers.meta import ProviderInfo
from app.providers.types import LLMChunk, LLMMessage, LLMToolCallDelta, ToolSpec


class OpenAILLMProvider:
    """Streaming chat-completions adapter for OpenAI-compatible APIs.

    Text deltas are forwarded as they stream. Tool calls arrive as fragmented
    deltas (indexed) and are reassembled across chunks before being emitted, so
    callers never see partial JSON arguments.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout_s = timeout_s

    async def close(self) -> None:
        return None

    def metadata(self) -> ProviderInfo:
        return ProviderInfo(
            name="openai",
            kind="llm",
            vendor="OpenAI",
            model=self.model,
            streaming=True,
            endpoint=self.base_url,
        )

    def _messages(self, messages: Sequence[LLMMessage]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for message in messages:
            if message.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": message.content,
                    }
                )
            elif message.role == "assistant":
                assistant: dict[str, object] = {"role": "assistant", "content": message.content}
                if message.tool_calls:
                    assistant["tool_calls"] = message.tool_calls
                out.append(assistant)
            else:
                out.append({"role": message.role, "content": message.content})
        return out

    def _tools(self, tools: Sequence[ToolSpec]) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]

    async def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[LLMChunk]:
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)
        request: dict[str, object] = {
            "model": self.model,
            "messages": self._messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = self._tools(tools)
        if temperature is not None:
            request["temperature"] = temperature

        accumulators: dict[int, dict[str, str]] = {}
        try:
            response = await client.chat.completions.create(**request)
            async for event in response:
                if not event.choices:
                    if event.usage:
                        yield LLMChunk(usage=dict(event.usage) if event.usage else None)
                    continue
                choice = event.choices[0]
                delta = choice.delta
                if delta and delta.content:
                    yield LLMChunk(text=delta.content)
                if delta and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        acc = accumulators.setdefault(tool_call.index, {"id": "", "name": "", "arguments": ""})
                        if tool_call.id:
                            acc["id"] += tool_call.id
                        if tool_call.function and tool_call.function.name:
                            acc["name"] += tool_call.function.name
                        if tool_call.function and tool_call.function.arguments:
                            acc["arguments"] += tool_call.function.arguments
                        if tool_call.function and tool_call.function.arguments:
                            yield LLMChunk(
                                tool_call=LLMToolCallDelta(
                                    index=tool_call.index,
                                    id=acc["id"] or None,
                                    name=acc["name"] or None,
                                    arguments=tool_call.function.arguments,
                                )
                            )
                if choice.finish_reason:
                    yield LLMChunk(finish_reason=choice.finish_reason)
        finally:
            await client.close()
