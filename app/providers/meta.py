from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """Self-describing metadata for a provider instance.

    Used by the benchmark to label latency results with exactly which provider,
    vendor, model and transport produced them -- turning provider choice from an
    opinion into a measured comparison.
    """

    name: str
    kind: str  # "stt" | "llm" | "tts"
    model: str | None = None
    vendor: str | None = None
    streaming: bool = False
    voice_id: str | None = None
    endpoint: str | None = None
    extra: dict[str, str] | None = None


def provider_metadata(obj: object) -> ProviderInfo | None:
    """Convenience accessor for instances exposing a ``metadata()`` method."""
    accessor = getattr(obj, "metadata", None)
    if callable(accessor):
        return accessor()
    return None
