from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DecisionKind = Literal["speech_start", "speech_end"]


@dataclass(slots=True)
class VadDecision:
    kind: DecisionKind
    energy_db: float | None = None
    speech_duration_ms: float | None = None
    timestamp: float = field(default_factory=time.time)


class EnergyVAD:
    """Frame-based energy voice activity detector with onset/offset hysteresis.

    The detector consumes arbitrarily sized chunks of mono float32 audio and
    reports confirmations, not raw thresholds: speech onset is confirmed only
    after ``start_confirm_ms`` of consecutive loud frames, and offset only after
    ``end_confirm_ms`` of consecutive quiet frames. Because the offset threshold
    sits below the onset threshold, brief dips inside a word do not terminate a
    segment.

    All configuration is derived from thresholds so endpointing behavior can be
    tuned without code changes.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        speech_start_rms: float = 0.015,
        speech_end_rms: float = 0.010,
        start_confirm_ms: int = 180,
        end_confirm_ms: int = 450,
    ) -> None:
        if speech_end_rms > speech_start_rms:
            raise ValueError("speech_end_rms must be <= speech_start_rms")
        if sample_rate <= 0 or frame_ms <= 0:
            raise ValueError("sample_rate and frame_ms must be positive")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.speech_start_rms = speech_start_rms
        self.speech_end_rms = speech_end_rms
        self.onset_frames = max(1, math.ceil(start_confirm_ms / frame_ms))
        self.offset_frames = max(1, math.ceil(end_confirm_ms / frame_ms))

        self._pending: np.ndarray = np.zeros(0, dtype=np.float32)
        self._speech_active = False
        self._onset_count = 0
        self._offset_count = 0
        self._speech_frames = 0
        self.total_speech_ms = 0.0

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def speech_duration_ms(self) -> float:
        return self._speech_frames * self.frame_ms

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech_active = False
        self._onset_count = 0
        self._offset_count = 0
        self._speech_frames = 0
        self.total_speech_ms = 0.0

    def _frame_rms(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))

    def process(self, samples: np.ndarray) -> list[VadDecision]:
        """Feed samples and return any confirmed speech_start / speech_end events."""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("EnergyVAD expects mono float32 samples")
        if samples.size:
            self._pending = np.concatenate([self._pending, samples])

        decisions: list[VadDecision] = []
        frame = self.frame_samples
        while self._pending.size >= frame:
            block = self._pending[:frame]
            self._pending = self._pending[frame:]
            rms_value = self._frame_rms(block)
            energy_db = 20.0 * math.log10(max(rms_value, 1e-9))

            if not self._speech_active:
                if rms_value >= self.speech_start_rms:
                    self._onset_count += 1
                    if self._onset_count >= self.onset_frames:
                        self._speech_active = True
                        self._onset_count = 0
                        self._speech_frames = 1
                        decisions.append(VadDecision("speech_start", energy_db=energy_db))
                else:
                    self._onset_count = 0
            else:
                self._speech_frames += 1
                if rms_value >= self.speech_end_rms:
                    self._offset_count = 0
                else:
                    self._offset_count += 1
                    if self._offset_count >= self.offset_frames:
                        self._speech_active = False
                        duration_ms = self._speech_frames * self.frame_ms
                        self.total_speech_ms += duration_ms
                        self._offset_count = 0
                        self._speech_frames = 0
                        decisions.append(VadDecision("speech_end", energy_db=energy_db, speech_duration_ms=duration_ms))
        return decisions

    def force_end(self) -> VadDecision | None:
        """Terminate an ongoing speech segment immediately (e.g. on disconnect)."""
        if not self._speech_active:
            return None
        self._speech_active = False
        duration_ms = self._speech_frames * self.frame_ms
        self.total_speech_ms += duration_ms
        self._offset_count = 0
        self._speech_frames = 0
        return VadDecision("speech_end", speech_duration_ms=duration_ms)
