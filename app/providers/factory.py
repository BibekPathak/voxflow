from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.providers.interfaces import LLMProvider, STTProvider, TTSProvider
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.openai import OpenAILLMProvider
from app.providers.meta import ProviderInfo
from app.providers.stt.deepgram import DeepgramSTTProvider
from app.providers.stt.mock import MockSTTProvider
from app.providers.tts.cartesia import CartesiaTTSProvider
from app.providers.tts.elevenlabs import ElevenLabsTTSProvider
from app.providers.tts.mock import MockTTSProvider
from app.runtime.errors import ConfigurationError, ProviderUnavailableError


def _info(provider: object) -> dict[str, object] | None:
    accessor = getattr(provider, "metadata", None)
    if callable(accessor):
        value = accessor()
        if isinstance(value, ProviderInfo):
            return {
                "name": value.name,
                "vendor": value.vendor,
                "model": value.model,
                "streaming": value.streaming,
                "voice_id": value.voice_id,
                "endpoint": value.endpoint,
            }
    return None


@dataclass(slots=True)
class ProviderSet:
    stt: STTProvider
    llm: LLMProvider
    tts: TTSProvider

    async def close(self) -> None:
        for provider in (self.stt, self.llm, self.tts):
            closer = getattr(provider, "close", None)
            if closer is not None:
                await closer()

    def describe(self) -> dict[str, dict[str, object] | None]:
        return {"stt": _info(self.stt), "llm": _info(self.llm), "tts": _info(self.tts)}

    @property
    def uses_real_providers(self) -> bool:
        return any((_info(p) or {}).get("name") not in ("mock", None) for p in (self.stt, self.llm, self.tts))


def _require(secret: str | None, name: str) -> str:
    if not secret:
        raise ProviderUnavailableError(
            provider=name,
            message=f"{name} is selected but no API key is configured (see .env.example)",
        )
    return secret


def _build_stt(settings: Settings) -> STTProvider:
    name = settings.provider_stt
    if name == "mock":
        return MockSTTProvider()
    if name == "deepgram":
        return DeepgramSTTProvider(
            _require(settings.deepgram_api_key.get_secret_value() if settings.deepgram_api_key else None, "deepgram"),
            model=settings.deepgram_model,
            endpoint=settings.deepgram_endpoint,
            timeout_s=settings.provider_stt_timeout_s,
        )
    raise ConfigurationError(f"unknown STT provider {name!r}")


def _build_llm(settings: Settings) -> LLMProvider:
    name = settings.provider_llm
    if name == "mock":
        return MockLLMProvider(
            first_token_ms=settings.mock_llm_first_token_ms,
            token_interval_ms=settings.mock_llm_token_interval_ms,
        )
    if name == "openai":
        return OpenAILLMProvider(
            _require(settings.openai_api_key.get_secret_value() if settings.openai_api_key else None, "openai"),
            model=settings.openai_llm_model,
            base_url=settings.openai_base_url,
            timeout_s=settings.provider_llm_timeout_s,
        )
    raise ConfigurationError(f"unknown LLM provider {name!r}")


def _build_tts(settings: Settings) -> TTSProvider:
    name = settings.provider_tts
    if name == "mock":
        return MockTTSProvider(first_audio_ms=settings.mock_tts_first_audio_ms)
    if name == "cartesia":
        voice_id = settings.cartesia_voice_id
        if not voice_id:
            raise ProviderUnavailableError(
                provider="cartesia",
                message="cartesia is selected but CARTESIA_VOICE_ID is not configured",
            )
        return CartesiaTTSProvider(
            _require(settings.cartesia_api_key.get_secret_value() if settings.cartesia_api_key else None, "cartesia"),
            voice_id=voice_id,
            model_id=settings.cartesia_model,
            url=settings.cartesia_ws_url,
            timeout_s=settings.provider_tts_timeout_s,
        )
    if name == "elevenlabs":
        voice_id = settings.elevenlabs_voice_id
        if not voice_id:
            raise ProviderUnavailableError(
                provider="elevenlabs",
                message="elevenlabs is selected but ELEVENLABS_VOICE_ID is not configured",
            )
        return ElevenLabsTTSProvider(
            _require(
                settings.elevenlabs_api_key.get_secret_value() if settings.elevenlabs_api_key else None,
                "elevenlabs",
            ),
            voice_id=voice_id,
            model_id=settings.elevenlabs_model,
            timeout_s=settings.provider_tts_timeout_s,
        )
    raise ConfigurationError(f"unknown TTS provider {name!r}")


def build_providers(settings: Settings) -> ProviderSet:
    """Build the configured provider set (mock by default; real adapters when
    the matching ``PROVIDER_*`` value is set and the key is present)."""
    return ProviderSet(stt=_build_stt(settings), llm=_build_llm(settings), tts=_build_tts(settings))


def provider_fingerprint(settings: Settings) -> dict[str, object]:
    """Non-secret configuration fingerprint recorded in benchmark reports so
    results can be reproduced and compared across runs. API keys are excluded."""

    def tts_voice(key: str | None) -> str:
        return f"<{key[:4]}...>" if key else None

    return {
        "provider_stt": settings.provider_stt,
        "stt_model": settings.deepgram_model if settings.provider_stt == "deepgram" else None,
        "provider_llm": settings.provider_llm,
        "llm_model": settings.openai_llm_model if settings.provider_llm == "openai" else None,
        "provider_tts": settings.provider_tts,
        "tts_model": settings.cartesia_model
        if settings.provider_tts == "cartesia"
        else (settings.elevenlabs_model if settings.provider_tts == "elevenlabs" else None),
        "tts_voice_id": tts_voice(
            settings.cartesia_voice_id if settings.provider_tts == "cartesia" else settings.elevenlabs_voice_id
        ),
        "sample_rate": settings.sample_rate,
        "turn_silence_ms": settings.turn_silence_ms,
        "turn_max_utterance_s": settings.turn_max_utterance_s,
    }
