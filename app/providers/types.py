from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Shared wire types exchanged with providers. The runtime and the adapters both
# depend only on these shapes, so a provider can be swapped without touching
# session logic.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Transcript:
    text: str
    is_final: bool = False
    words: list[dict[str, Any]] | None = None
    language: str | None = None


@dataclass(slots=True)
class LLMMessage:
    role: str
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict[str, Any] | None = None


@dataclass(slots=True)
class LLMToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(slots=True)
class LLMChunk:
    text: str = ""
    tool_call: LLMToolCallDelta | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


@dataclass(slots=True)
class AudioData:
    pcm: bytes = b""
    sample_rate: int = 16_000

    @property
    def duration_ms(self) -> float:
        if self.sample_rate <= 0 or not self.pcm:
            return 0.0
        return len(self.pcm) / 2 / self.sample_rate * 1000.0
