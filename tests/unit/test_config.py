from __future__ import annotations

from app.config import Settings


def test_default_settings() -> None:
    settings = Settings()
    assert settings.app_name == "voxflow"
    assert settings.sample_rate == 16_000
    assert settings.audio_frame_ms == 20
    assert settings.provider_stt == "mock"
    assert settings.provider_llm == "mock"
    assert settings.provider_tts == "mock"
    assert settings.barge_in_enabled is True
    assert settings.database_url.startswith("sqlite+aiosqlite")


def test_secret_fields_are_optional() -> None:
    settings = Settings()
    assert settings.deepgram_api_key is None
    assert settings.openai_api_key is None
    assert settings.cartesia_api_key is None


def test_settings_accept_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SAMPLE_RATE", "8000")
    monkeypatch.setenv("PROVIDER_STT", "deepgram")
    settings = Settings()
    assert settings.sample_rate == 8_000
    assert settings.provider_stt == "deepgram"
