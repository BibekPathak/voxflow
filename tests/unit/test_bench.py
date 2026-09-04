from __future__ import annotations

from app.evaluation.bench import (
    DEFAULT_SCRIPT,
    _generate_prompt_audio,
    _verdict,
    bench_report_markdown,
    run_benchmark,
    run_mock_baseline,
    save_report,
    save_report_markdown,
)
from app.evaluation.support import SessionHarness
from app.providers.meta import ProviderInfo
from app.providers.types import AudioData
from app.runtime.events import ToolCallStarted


class FakeTTS:
    """In-memory TTS that emits deterministic PCM, to test prompt caching."""

    def __init__(self) -> None:
        self.calls = 0

    def metadata(self) -> ProviderInfo:
        return ProviderInfo(name="fake", kind="tts", vendor="Fake", model="v0", streaming=True)

    async def synthesize(self, text: str, *, sample_rate: int = 16_000) -> AudioData:
        del text
        self.calls += 1
        return AudioData(pcm=b"\x00\x00" * 6400, sample_rate=sample_rate)


async def test_prompt_audio_is_cached(tmp_path, monkeypatch) -> None:
    import app.evaluation.bench as bench

    monkeypatch.setattr(bench, "_PCM_DIR", tmp_path)
    tts = FakeTTS()
    first = await _generate_prompt_audio(tts, "hello world", index=0, sample_rate=16_000, force=False)
    second = await _generate_prompt_audio(tts, "hello world", index=0, sample_rate=16_000, force=False)
    assert first == second
    assert tts.calls == 1  # cached, second call did not re-synthesize


async def test_prompt_audio_cache_forces_regeneration(tmp_path, monkeypatch) -> None:
    import app.evaluation.bench as bench

    monkeypatch.setattr(bench, "_PCM_DIR", tmp_path)
    tts = FakeTTS()
    await _generate_prompt_audio(tts, "hello world", index=0, sample_rate=16_000, force=False)
    await _generate_prompt_audio(tts, "hello world", index=0, sample_rate=16_000, force=True)
    assert tts.calls == 2


def test_verdict_logic() -> None:
    inspected = [ToolCallStarted(session_id="s", turn_id=1, tool_name="inspect_payment")]
    assert _verdict(prompt_text="why was my payment pay_101 declined", tool_starts=inspected) is True
    assert _verdict(prompt_text="why was my payment declined", tool_starts=[]) is False
    assert _verdict(prompt_text="what fees do you charge", tool_starts=[]) is None


async def test_mock_baseline_returns_expected_keys() -> None:
    baseline = await run_mock_baseline(turns=3)
    assert set(baseline) == {"ttft", "ttfa", "e2e"}
    assert all(v >= 0 for v in baseline.values())


async def test_benchmark_runs_with_injected_runtime_and_report(tmp_path, monkeypatch) -> None:
    import numpy as np

    from app.audio.resampling import float32_to_pcm16_bytes
    from app.evaluation import bench
    from app.providers.factory import ProviderSet
    from app.providers.llm.mock import MockLLMProvider
    from app.providers.stt.mock import MockSTTProvider

    monkeypatch.setattr(bench, "_PCM_DIR", tmp_path)
    monkeypatch.setattr(bench, "_REPORT_DIR", tmp_path)

    class VoicedTTS:
        def metadata(self):
            return ProviderInfo(name="voiced", kind="tts", vendor="X", model="v", streaming=True)

        async def synthesize(self, text, *, sample_rate=16000):
            del text
            n = int(0.5 * sample_rate)
            wave = (0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / sample_rate)).astype("float32")
            return AudioData(pcm=float32_to_pcm16_bytes(wave), sample_rate=sample_rate)

    harness = SessionHarness()
    await harness.open()
    runtime = harness.runtime
    runtime.providers = ProviderSet(
        stt=MockSTTProvider(utterances=["what fees do you charge"]),
        llm=MockLLMProvider(),
        tts=VoicedTTS(),
    )

    report = await run_benchmark(
        turns=2,
        runtime_factory=lambda: _coroutine_returning(runtime),
        close_runtime=_noop_close,
        tts_override=VoicedTTS(),
        baseline_mock={"ttft": 1.0, "ttfa": 20.0, "e2e": 344.0},
    )
    assert report.turns == 2
    assert report.results
    assert report.provider_description["stt"] is not None

    md = bench_report_markdown(report)
    assert "Real Provider Benchmark" in md
    assert "P50" in md and "P95" in md
    assert "Mock vs Real" in md

    json_path = save_report(report, directory=tmp_path)
    md_path = save_report_markdown(report, directory=tmp_path)
    assert json_path.exists()
    assert md_path.exists()

    await harness.close()


def test_default_script_has_reasonable_length() -> None:
    assert len(DEFAULT_SCRIPT) >= 4
    assert all(isinstance(p, str) and p for p in DEFAULT_SCRIPT)


async def _coroutine_returning(runtime):
    return runtime


async def _noop_close(runtime) -> None:
    return None
