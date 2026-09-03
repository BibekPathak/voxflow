from __future__ import annotations

from app.config import Settings
from app.providers.factory import build_providers
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.openai import OpenAILLMProvider
from app.providers.stt.deepgram import DeepgramSTTProvider
from app.providers.stt.mock import MockSTTProvider
from app.providers.tts.cartesia import CartesiaTTSProvider
from app.providers.tts.elevenlabs import ElevenLabsTTSProvider
from app.providers.tts.mock import MockTTSProvider
from app.runtime.errors import ProviderUnavailableError
from tests.conftest import make_settings


def _settings(**overrides: object) -> Settings:
    return make_settings(**overrides)


async def test_factory_defaults_to_mocks() -> None:
    providers = build_providers(_settings())
    assert isinstance(providers.stt, MockSTTProvider)
    assert isinstance(providers.llm, MockLLMProvider)
    assert isinstance(providers.tts, MockTTSProvider)
    await providers.close()


async def test_deepgram_selected_without_key_fails() -> None:
    try:
        build_providers(_settings(provider_stt="deepgram"))
    except ProviderUnavailableError as exc:
        assert exc.provider == "deepgram"
    else:
        raise AssertionError("expected ProviderUnavailableError")


async def test_openai_selected_without_key_fails() -> None:
    try:
        build_providers(_settings(provider_llm="openai"))
    except ProviderUnavailableError as exc:
        assert exc.provider == "openai"
    else:
        raise AssertionError("expected ProviderUnavailableError")


async def test_cartesia_selected_without_key_or_voice_fails() -> None:
    try:
        build_providers(_settings(provider_tts="cartesia"))
    except ProviderUnavailableError as exc:
        assert exc.provider == "cartesia"
    else:
        raise AssertionError("expected ProviderUnavailableError")


async def test_real_adapter_construction_is_offline_safe() -> None:
    stt = DeepgramSTTProvider("test-key")
    llm = OpenAILLMProvider("test-key", model="gpt-4o-mini")
    tts_c = CartesiaTTSProvider("test-key", voice_id="voice_1")
    tts_e = ElevenLabsTTSProvider("test-key", voice_id="voice_1")
    assert stt.model == "nova-3"
    assert llm.model == "gpt-4o-mini"
    assert tts_c.voice_id == "voice_1"
    assert tts_e.voice_id == "voice_1"
    for provider in (stt, llm, tts_c, tts_e):
        await provider.close()


def test_deepgram_listen_url_contains_expected_params() -> None:
    provider = DeepgramSTTProvider("k", model="nova-3")
    url = provider._listen_url(sample_rate=16_000, interim_results=True, language="en-US")
    assert "encoding=linear16" in url
    assert "sample_rate=16000" in url
    assert "interim_results=true" in url
    assert "model=nova-3" in url
    assert "language=en-US" in url
