from __future__ import annotations

import numpy as np

from app.audio.buffering import AudioBuffer, SequenceGapDetector
from app.audio.gateway import AudioGateway
from tests.unit.test_vad import silence_samples, tone_samples


def test_audio_buffer_accumulates_and_drains() -> None:
    buf = AudioBuffer(sample_rate=16_000)
    buf.append(tone_samples(0.1, amplitude=0.2))
    buf.append(silence_samples(0.1))
    assert buf.num_samples == 3_200
    assert abs(buf.duration_ms - 200) < 1.0
    drained = buf.drain()
    assert len(drained) == 3_200
    assert buf.num_samples == 0


def test_sequence_gap_detector_counts_drops() -> None:
    detector = SequenceGapDetector()
    detector.observe(0)
    detector.observe(1)
    detector.observe(4)
    detector.observe(5)
    assert detector.missing_frames == 2
    assert detector.total_gaps == 1


def test_sequence_gap_detector_ignores_duplicates() -> None:
    detector = SequenceGapDetector()
    detector.observe(3)
    detector.observe(3)
    detector.observe(4)
    assert detector.total_gaps == 0


def test_gateway_ingest_reports_decision_and_rms() -> None:
    gw = AudioGateway(sample_rate=16_000, vad_start_confirm_ms=40, vad_end_confirm_ms=80)
    audio = np.concatenate([tone_samples(0.25, amplitude=0.3), silence_samples(0.3)])
    result = gw.ingest_pcm((audio * 32000).astype(np.int16).tobytes())
    assert result.rms > 0.1
    assert [d.kind for d in result.decisions] == ["speech_start", "speech_end"]
    assert gw.total_samples_in == audio.size


def test_gateway_sequence_assignment() -> None:
    gw = AudioGateway(sample_rate=16_000)
    assert gw.next_sequence() == 0
    assert gw.next_sequence() == 1


def test_gateway_empty_ingest_is_noop() -> None:
    gw = AudioGateway(sample_rate=16_000)
    result = gw.ingest_pcm(b"")
    assert result.num_samples == 0
    assert result.decisions == []
