from __future__ import annotations

from dataclasses import dataclass

from app.audio.buffering import AudioBuffer, SequenceGapDetector
from app.audio.resampling import pcm16_bytes_to_float32, rms, to_mono_float32
from app.audio.vad import EnergyVAD, VadDecision


@dataclass(slots=True)
class IngestResult:
    num_samples: int
    rms: float
    decisions: list[VadDecision]


class AudioGateway:
    """Ingestion front-end for a single session's inbound audio.

    Owns PCM decoding, mono conversion, jitter accounting, and the energy VAD.
    The gateway is deliberately dumb about conversation semantics: it turns raw
    bytes into float samples plus confirmed speech-start/end decisions and hands
    those up to the session runtime, which decides what they mean.

    A gap detector is wired in so network-degraded audio (dropped chunks) can be
    observed rather than silently corrupting endpointing decisions.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        expected_channels: int = 1,
        capture_audio: bool = False,
        vad_speech_start_rms: float = 0.015,
        vad_speech_end_rms: float = 0.010,
        vad_start_confirm_ms: int = 180,
        vad_end_confirm_ms: int = 450,
        vad_frame_ms: int = 20,
    ) -> None:
        self.sample_rate = sample_rate
        self.expected_channels = expected_channels
        self.capture_audio = capture_audio
        self.vad = EnergyVAD(
            sample_rate=sample_rate,
            frame_ms=vad_frame_ms,
            speech_start_rms=vad_speech_start_rms,
            speech_end_rms=vad_speech_end_rms,
            start_confirm_ms=vad_start_confirm_ms,
            end_confirm_ms=vad_end_confirm_ms,
        )
        self.recording = AudioBuffer(sample_rate)
        self.sequencer = SequenceGapDetector()
        self._sequence = 0
        self.total_samples_in = 0

    def ingest_pcm(self, data: bytes, *, sequence: int | None = None) -> IngestResult:
        if not data:
            return IngestResult(num_samples=0, rms=0.0, decisions=[])
        if sequence is not None:
            self.sequencer.observe(sequence)
        mono = to_mono_float32(pcm16_bytes_to_float32(data), channels=self.expected_channels)
        if mono.size == 0:
            return IngestResult(num_samples=0, rms=0.0, decisions=[])
        self.total_samples_in += mono.size
        if self.capture_audio:
            self.recording.append(mono)
        level = rms(mono)
        decisions = self.vad.process(mono)
        return IngestResult(num_samples=int(mono.size), rms=level, decisions=decisions)

    def next_sequence(self) -> int:
        seq = self._sequence
        self._sequence += 1
        return seq

    def reset_turn(self) -> None:
        self.recording.clear()
        self.sequencer.reset()
        self._sequence = 0
