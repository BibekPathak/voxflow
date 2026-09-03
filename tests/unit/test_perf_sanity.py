from __future__ import annotations

import math
import time

from app.audio.gateway import AudioGateway
from app.audio.resampling import float32_to_pcm16_bytes
from tests.unit.test_vad import silence_samples, tone_samples


def _chunks(seconds_total: float, chunk_seconds: float = 0.256) -> list[bytes]:
    count = int(math.ceil(seconds_total / chunk_seconds))
    chunks: list[bytes] = []
    speech = float32_to_pcm16_bytes(tone_samples(chunk_seconds, amplitude=0.3))
    silence = float32_to_pcm16_bytes(silence_samples(chunk_seconds))
    for index in range(count):
        chunks.append(speech if index % 5 < 3 else silence)
    return chunks


def test_gateway_processes_audio_many_times_faster_than_realtime() -> None:
    gateway = AudioGateway(
        vad_speech_start_rms=0.01,
        vad_speech_end_rms=0.005,
        vad_start_confirm_ms=40,
        vad_end_confirm_ms=80,
    )
    chunks = _chunks(seconds_total=128)
    started = time.monotonic()
    decisions = 0
    for chunk in chunks:
        decisions += len(gateway.ingest_pcm(chunk).decisions)
    elapsed = time.monotonic() - started
    assert decisions > 0
    # ~128s of audio must be processed far faster than real time; the bound is
    # intentionally loose so the check never flakes on loaded CI machines.
    assert elapsed < 8.0
    assert gateway.total_samples_in == len(chunks) * len(chunks[0]) // 2


def test_gateway_does_not_buffer_unbounded_recording_by_default() -> None:
    gateway = AudioGateway()
    chunks = _chunks(seconds_total=8)
    for chunk in chunks:
        gateway.ingest_pcm(chunk)
    assert gateway.recording.num_samples == 0
    assert gateway.total_samples_in == len(chunks) * len(chunks[0]) // 2


def test_gateway_recording_can_be_enabled() -> None:
    gateway = AudioGateway(capture_audio=True)
    chunks = _chunks(seconds_total=2)
    for chunk in chunks:
        gateway.ingest_pcm(chunk)
    assert gateway.recording.num_samples == len(chunks) * len(chunks[0]) // 2
