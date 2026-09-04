from __future__ import annotations

from collections.abc import AsyncIterator

from app.audio.resampling import pcm16_bytes_to_float32, rms
from app.providers.meta import ProviderInfo
from app.providers.types import Transcript

DEFAULT_SCRIPT = [
    "my payment failed yesterday",
    "no I do not have the transaction id",
]


class MockSTTProvider:
    """Offline, deterministic streaming STT.

    Instead of calling a speech model, the mock recognizes *voiced* audio the
    same way the energy VAD does (RMS over a chunk with onset/offset hangover)
    and, for every detected utterance, plays back the next utterance from a
    fixed script as streaming partials followed by a final transcript. This makes
    the entire audio -> STT path exercisable in tests and local demos without an
    API key or network.

    Utterances advance in order across the whole stream call, matching a
    multi-turn voice conversation where the human reads each scripted line.
    """

    def __init__(
        self,
        utterances: list[str] | None = None,
        *,
        speech_start_rms: float = 0.02,
        silence_chunks: int = 4,
        voiced_start_chunks: int = 2,
    ) -> None:
        self.utterances = list(utterances or DEFAULT_SCRIPT)
        self.speech_start_rms = speech_start_rms
        self.silence_chunks = max(1, silence_chunks)
        self.voiced_start_chunks = max(1, voiced_start_chunks)
        self._script_index = 0

    def reset(self) -> None:
        self._script_index = 0

    async def close(self) -> None:
        return None

    def metadata(self) -> ProviderInfo:
        return ProviderInfo(name="mock", kind="stt", vendor="VoxFlow Mock", model="scripted", streaming=True)

    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        sample_rate: int = 16_000,
        interim_results: bool = True,
        language: str | None = None,
    ) -> AsyncIterator[Transcript]:
        del sample_rate
        end_rms = self.speech_start_rms * 0.5

        target: str | None = None
        speech = False
        voiced_run = 0
        silent_run = 0
        shown_len = 0
        last_partial = ""

        async for chunk in audio:
            if not chunk:
                continue
            samples = pcm16_bytes_to_float32(chunk)
            if samples.size == 0:
                continue
            level = rms(samples)

            if speech:
                if target is None:
                    speech = False
                    continue
                if level >= end_rms:
                    silent_run = 0
                    shown_len = min(len(target), shown_len + 1)
                    partial = target[:shown_len]
                    if partial != last_partial and partial and interim_results:
                        yield Transcript(text=partial, is_final=False, language=language)
                        last_partial = partial
                else:
                    silent_run += 1
                    if silent_run >= self.silence_chunks:
                        if target:
                            yield Transcript(text=target, is_final=True, language=language)
                        speech = False
                        target = None
                        shown_len = 0
                        last_partial = ""
                        silent_run = 0
            else:
                if level >= self.speech_start_rms:
                    voiced_run += 1
                    if voiced_run >= self.voiced_start_chunks and self._script_index < len(self.utterances):
                        target = self.utterances[self._script_index]
                        self._script_index += 1
                        speech = True
                        voiced_run = 0
                        shown_len = 0
                else:
                    voiced_run = 0

        if speech and target is not None:
            yield Transcript(text=target, is_final=True, language=language)
