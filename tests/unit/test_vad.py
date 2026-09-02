from __future__ import annotations

import math

import numpy as np

from app.audio.resampling import (
    float32_to_pcm16_bytes,
    pcm16_bytes_to_float32,
    resample_linear,
    rms,
    to_mono_float32,
)
from app.audio.vad import EnergyVAD


def tone_samples(seconds: float, amplitude: float, sample_rate: int = 16_000, freq: float = 440.0) -> np.ndarray:
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def silence_samples(seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def make_vad() -> EnergyVAD:
    return EnergyVAD(
        sample_rate=16_000,
        frame_ms=20,
        speech_start_rms=0.01,
        speech_end_rms=0.005,
        start_confirm_ms=40,
        end_confirm_ms=80,
    )


def test_detects_speech_burst() -> None:
    vad = make_vad()
    audio = np.concatenate(
        [
            silence_samples(0.2),
            tone_samples(0.3, amplitude=0.3),
            silence_samples(0.3),
        ]
    )
    decisions = vad.process(audio)
    kinds = [d.kind for d in decisions]
    assert kinds == ["speech_start", "speech_end"]
    assert not vad.speech_active


def test_detects_speech_in_two_chunks() -> None:
    vad = make_vad()
    first = vad.process(silence_samples(0.1))
    assert first == []
    assert not vad.speech_active
    second = vad.process(tone_samples(0.15, amplitude=0.3))
    assert second and second[0].kind == "speech_start"
    third = vad.process(silence_samples(0.3))
    assert third and third[0].kind == "speech_end"


def test_silence_produces_no_decisions() -> None:
    vad = make_vad()
    assert vad.process(silence_samples(0.5)) == []


def test_low_amplitude_noise_produces_no_decisions() -> None:
    vad = make_vad()
    noise = 0.002 * np.random.default_rng(1).standard_normal(16_000 // 2).astype(np.float32)
    assert vad.process(noise) == []


def test_short_dip_inside_speech_does_not_end_segment() -> None:
    vad = make_vad()
    dip = np.concatenate(
        [
            tone_samples(0.2, amplitude=0.3),
            silence_samples(0.04),
            tone_samples(0.2, amplitude=0.3),
            silence_samples(0.3),
        ]
    )
    decisions = vad.process(dip)
    assert [d.kind for d in decisions] == ["speech_start", "speech_end"]
    assert not vad.speech_active


def test_speech_duration_reported() -> None:
    vad = make_vad()
    speech = tone_samples(0.4, amplitude=0.3)
    audio = np.concatenate([silence_samples(0.1), speech, silence_samples(0.3)])
    decisions = vad.process(audio)
    end = decisions[-1]
    assert 300 <= (end.speech_duration_ms or 0) <= 500


def test_rms_of_sine_is_amplitude_over_sqrt2() -> None:
    samples = tone_samples(0.1, amplitude=0.5)
    assert abs(rms(samples) - 0.5 / math.sqrt(2)) < 0.01


def test_force_end_returns_decision() -> None:
    vad = make_vad()
    vad.process(np.concatenate([silence_samples(0.1), tone_samples(0.3, amplitude=0.3)]))
    assert vad.speech_active
    decision = vad.force_end()
    assert decision is not None
    assert decision.kind == "speech_end"
    assert not vad.speech_active
    assert vad.force_end() is None


def test_pcm_roundtrip() -> None:
    samples = tone_samples(0.05, amplitude=0.4)
    pcm = float32_to_pcm16_bytes(samples)
    decoded = pcm16_bytes_to_float32(pcm)
    assert np.allclose(decoded, samples, atol=1 / 32_768)


def test_resample_linear_doubles_length() -> None:
    samples = tone_samples(0.1, amplitude=0.3, sample_rate=8_000)
    resampled = resample_linear(samples, 8_000, 16_000)
    assert len(resampled) == 2 * len(samples)


def test_to_mono_averages_channels() -> None:
    stereo = np.array([[0.5, -0.5], [0.2, 0.2], [0.0, 0.0]], dtype=np.float32)
    mono = to_mono_float32(stereo, channels=2)
    assert np.allclose(mono, [0.0, 0.2, 0.0])
