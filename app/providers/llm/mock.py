from __future__ import annotations

import asyncio
import itertools
import re
from collections.abc import AsyncIterator, Sequence

from app.providers.types import LLMChunk, LLMMessage, LLMToolCallDelta, ToolSpec

_TOKEN_SPLIT = re.compile(r"\S+\s*")

_TOOL_KEYWORDS: dict[str, str] = {
    "customer": "search_customer",
    "account": "search_customer",
    "transaction": "get_recent_transactions",
    "payment": "inspect_payment",
    "declined": "inspect_payment",
    "ticket": "create_support_ticket",
    "refund": "create_support_ticket",
    "report": "create_support_ticket",
    "issue": "create_support_ticket",
}


class MockLLMProvider:
    """Offline, deterministic streaming LLM.

    The mock is rule-based so the full pipeline (including tool calling and
    barge-in cancellation of in-flight generation) is testable without a key:

    * if the conversation already contains a ``tool`` result, it answers from
      that result;
    * otherwise, if tools are offered and the user text mentions a known topic,
      it emits a single tool call for the matching tool;
    * otherwise it streams a canned textual reply token by token.

    All latency is simulated through ``first_token_ms`` / ``token_interval_ms``
    so time-to-first-token metrics behave realistically under mocks too.
    """

    def __init__(self, *, first_token_ms: float = 0.0, token_interval_ms: float = 0.0) -> None:
        self.first_token_ms = first_token_ms
        self.token_interval_ms = token_interval_ms
        self._calls = itertools.count(1)

    async def close(self) -> None:
        return None

    def _pick_tool(self, text: str, tools: Sequence[ToolSpec]) -> str | None:
        offered = {tool.name for tool in tools}
        lowered = text.lower()
        for keyword in sorted(_TOOL_KEYWORDS, key=len, reverse=True):
            tool_name = _TOOL_KEYWORDS[keyword]
            if keyword in lowered and tool_name in offered:
                return tool_name
        return None

    def _answer(self, messages: Sequence[LLMMessage], tool_result: bool) -> str:
        if tool_result:
            last_tool = next((m for m in reversed(messages) if m.role == "tool"), None)
            snippet = (last_tool.content or "").strip()[:200] if last_tool else ""
            if snippet:
                return f"I looked that up for you. Here is what I found: {snippet}"
            return "I looked that up for you, but I did not get a usable result."
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        text = (last_user.content if last_user else "").strip().lower()
        if not text:
            return "Sorry, I did not catch that. Could you repeat it?"
        if any(word in text for word in ("hi", "hello", "hey")):
            return "Hello! I can help you with payments, transactions, and support tickets. What would you like to do?"
        if "thank" in text:
            return "You're welcome! Is there anything else I can help you with?"
        if "what is" in text or "who" in text or "how" in text:
            return "Good question. Let me think about that for you."
        return "Sure, I can help with that. Could you tell me a bit more detail?"

    async def _stream_text(self, text: str) -> AsyncIterator[LLMChunk]:
        if self.first_token_ms:
            await asyncio.sleep(self.first_token_ms / 1000.0)
        first = True
        for token in _TOKEN_SPLIT.findall(text):
            if not first and self.token_interval_ms:
                await asyncio.sleep(self.token_interval_ms / 1000.0)
            yield LLMChunk(text=token)
            first = False
        yield LLMChunk(finish_reason="stop")

    async def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[ToolSpec] | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[LLMChunk]:
        del temperature
        has_tool_result = any(m.role == "tool" for m in messages)
        if tools and not has_tool_result:
            last_user = next((m for m in reversed(messages) if m.role == "user"), None)
            user_text = last_user.content if last_user else ""
            chosen = self._pick_tool(user_text, tools)
            if chosen is not None:
                if self.first_token_ms:
                    await asyncio.sleep(self.first_token_ms / 1000.0)
                call_id = f"call_mock_{next(self._calls)}"
                yield LLMChunk(tool_call=LLMToolCallDelta(index=0, id=call_id, name=chosen, arguments="{}"))
                yield LLMChunk(finish_reason="tool_calls")
                return

        answer = self._answer(messages, tool_result=has_tool_result)
        async for chunk in self._stream_text(answer):
            yield chunk
