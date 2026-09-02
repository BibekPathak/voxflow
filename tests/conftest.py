from __future__ import annotations

from app.config import Settings


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = dict(
        sample_rate=16_000,
        vad_speech_start_rms=0.01,
        vad_speech_end_rms=0.005,
        vad_start_confirm_ms=40,
        vad_end_confirm_ms=80,
        vad_frame_ms=20,
        turn_min_speech_ms=80,
        turn_silence_ms=120,
        turn_max_utterance_s=5,
        audio_queue_maxsize=256,
        provider_stt="mock",
        provider_llm="mock",
        provider_tts="mock",
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]
