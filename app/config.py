from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app import __version__


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "voxflow"
    app_version: str = __version__
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    json_logs: bool = False
    cors_origins: list[str] = ["*"]

    sample_rate: int = 16_000
    audio_frame_ms: int = 20
    audio_queue_maxsize: int = 1024

    vad_sensitivity: Literal["mock", "energy"] = "energy"
    vad_speech_start_rms: float = 0.015
    vad_speech_end_rms: float = 0.010
    vad_start_confirm_ms: int = 180
    vad_end_confirm_ms: int = 450
    vad_frame_ms: int = 20

    turn_silence_ms: int = 750
    turn_min_speech_ms: int = 400
    turn_max_utterance_s: int = 30
    turn_auto_submit_ms: int = 2_500
    turn_pause_window_ms: int = 600

    barge_in_enabled: bool = True
    barge_in_speech_start_rms: float = 0.02
    barge_in_cancel_tts: bool = True
    barge_in_cancel_llm: bool = True

    provider_stt: Literal["mock", "deepgram"] = "mock"
    provider_llm: Literal["mock", "openai"] = "mock"
    provider_tts: Literal["mock", "cartesia", "elevenlabs"] = "mock"

    bench_tts: Literal["cartesia", "elevenlabs"] = "cartesia"

    mock_llm_first_token_ms: float = 0.0
    mock_llm_token_interval_ms: float = 0.0
    mock_tts_first_audio_ms: float = 0.0

    deepgram_api_key: SecretStr | None = None
    deepgram_model: str = "nova-3"
    deepgram_endpoint: str = "wss://api.deepgram.com/v1/listen"

    openai_api_key: SecretStr | None = None
    openai_llm_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None

    cartesia_api_key: SecretStr | None = None
    cartesia_model: str = "sonic-2"
    cartesia_voice_id: str | None = None
    cartesia_ws_url: str = "wss://api.cartesia.ai/tts/websocket"

    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model: str = "eleven_turbo_v2_5"

    provider_stt_timeout_s: float = 15.0
    provider_llm_timeout_s: float = 30.0
    provider_tts_timeout_s: float = 20.0
    provider_max_retries: int = 2

    tool_timeout_s: float = 10.0
    tool_max_retries: int = 1

    context_max_turns: int = 10
    context_max_chars: int = 8_000
    context_summary_enabled: bool = False

    database_url: str = Field(
        default="sqlite+aiosqlite:///./voxflow.db",
        description="SQLAlchemy async URL. Docker Compose overrides with Postgres.",
    )
    redis_url: str = "redis://localhost:6379/0"
    session_state_backend: Literal["memory", "redis"] = "memory"
    persist_conversations: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
