from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.providers.interfaces import LLMProvider, STTProvider, TTSProvider
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.openai import OpenAILLMProvider
from app.providers.stt.deepgram import DeepgramSTTProvider
from app.providers.stt.mock import MockSTTProvider
from app.providers.tts.cartesia import CartesiaTTSProvider
from app.providers.tts.elevenlabs import ElevenLabsTTSProvider
from app.providers.tts.mock import MockTTSProvider
from app.runtime.errors import ConfigurationError, ProviderUnavailableError


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
