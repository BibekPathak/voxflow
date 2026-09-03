from __future__ import annotations

from collections.abc import Sequence

from app.memory.conversation import HistoryEntry
from app.providers.types import LLMMessage


def _trim_content(content: str, budget: int) -> str:
    if len(content) <= budget:
        return content
    return content[:budget].rstrip() + "…"


def build_context_messages(
    history: Sequence[HistoryEntry],
    *,
    system_prompt: str,
    user_text: str,
    max_turns: int = 10,
    max_chars: int = 8000,
) -> list[LLMMessage]:
    """Turn short-term history into an LLM prompt with a bounded context.

    A sliding window keeps the most recent ``max_turns`` messages and drops the
    oldest first until the total history stays under ``max_chars`` -- the LLM is
    never given an unbounded transcript. The current user utterance is always
    included last and is never truncated away.
    """
    recent = history[-max_turns:] if max_turns > 0 else []

    budget = max_chars - len(user_text)
    window: list[HistoryEntry] = []
    for entry in reversed(recent):
        needed = len(entry.content)
        if budget < needed and window:
            break
        if budget >= needed:
            window.append(entry)
            budget -= needed
        else:
            break
    window = list(reversed(window))

    messages = [LLMMessage(role="system", content=system_prompt)]
    for entry in window:
        messages.append(
            LLMMessage(
                role=entry.role if entry.role in ("user", "assistant") else "user",
                content=_trim_content(entry.content, max_chars),
            )
        )
    messages.append(LLMMessage(role="user", content=user_text))
    return messages
