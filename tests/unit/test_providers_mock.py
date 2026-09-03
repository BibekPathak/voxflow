from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from app.audio.resampling import float32_to_pcm16_bytes
from app.providers.llm.mock import MockLLMProvider
from app.providers.stt.mock import MockSTTProvider
from app.providers.tts.mock import MockTTSProvider
from app.providers.types import AudioData, LLMMessage, ToolSpec
from tests.unit.test_vad import silence_samples, tone_samples


def _pcm(samples: np.ndarray) -> bytes:
    return float32_to_pcm16_bytes(samples)


def _burst_audio(utterance_seconds: float, gaps: int = 1) -> list[bytes]:
    frame = 320
    chunks: list[bytes] = []

    def append_frames(samples: np.ndarray) -> None:
        for start in range(0, samples.size, frame):
            chunks.append(_pcm(samples[start : start + frame]))

    for _ in range(gaps + 1):
        append_frames(tone_samples(utterance_seconds, amplitude=0.3))
        append_frames(silence_samples(0.25))
    return chunks


async def _iter_bytes(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def _iter_text(parts: list[str]) -> AsyncIterator[str]:
    for part in parts:
        yield part


async def test_mock_stt_streams_partials_then_final() -> None:
    provider = MockSTTProvider(utterances=["my payment failed"])
    transcripts = [transcript async for transcript in provider.transcribe_stream(_iter_bytes(_burst_audio(0.5)))]
    assert transcripts[-1].text == "my payment failed"
    assert transcripts[-1].is_final is True
    partials = [t for t in transcripts if not t.is_final]
    assert partials
    assert partials[0].text != transcripts[-1].text
    assert len(partials[0].text) < len(transcripts[-1].text)
    assert all(
        partials[i].text == partials[i + 1].text or len(partials[i].text) < len(partials[i + 1].text)
        for i in range(len(partials) - 1)
    )


async def test_mock_stt_handles_two_utterances_in_order() -> None:
    provider = MockSTTProvider(utterances=["first question", "second question"])
    finals: list[str] = []
    async for transcript in provider.transcribe_stream(_iter_bytes(_burst_audio(0.4, gaps=2))):
        if transcript.is_final:
            finals.append(transcript.text)
    assert finals == ["first question", "second question"]


async def test_mock_stt_ignores_silence() -> None:
    provider = MockSTTProvider(utterances=["hello"])
    chunks = [_pcm(silence_samples(0.5))]
    transcripts = [t async for t in provider.transcribe_stream(_iter_bytes(chunks))]
    assert transcripts == []


async def test_mock_llm_plain_text_stream() -> None:
    provider = MockLLMProvider()
    chunks = [chunk async for chunk in provider.stream_chat([LLMMessage(role="user", content="hello there")])]
    text = "".join(chunk.text for chunk in chunks if chunk.text)
    assert "Hello!" in text
    assert chunks[-1].finish_reason == "stop"


async def test_mock_llm_selects_tool_by_keyword() -> None:
    provider = MockLLMProvider()
    tools = [
        ToolSpec(name="get_recent_transactions"),
        ToolSpec(name="inspect_payment", description="look up a payment"),
    ]
    chunks = [
        chunk
        async for chunk in provider.stream_chat(
            [LLMMessage(role="user", content="why was my payment declined")], tools=tools
        )
    ]
    tool_calls = [c.tool_call for c in chunks if c.tool_call]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "inspect_payment"
    assert tool_calls[0].id and tool_calls[0].id.startswith("call_mock_")
    assert chunks[-1].finish_reason == "tool_calls"


async def test_mock_llm_ignores_unoffered_tool() -> None:
    provider = MockLLMProvider()
    tools = [ToolSpec(name="search_customer")]
    chunks = [
        chunk
        async for chunk in provider.stream_chat(
            [LLMMessage(role="user", content="create a refund ticket")], tools=tools
        )
    ]
    assert not any(c.tool_call for c in chunks)
    assert any(c.text for c in chunks)


async def test_mock_llm_answers_tool_result() -> None:
    provider = MockLLMProvider()
    messages = [
        LLMMessage(role="user", content="check my payment"),
        LLMMessage(role="tool", tool_call_id="call_1", content="status=declined"),
    ]
    chunks = [chunk async for chunk in provider.stream_chat(messages, tools=None)]
    text = "".join(c.text for c in chunks if c.text)
    assert "declined" in text


async def test_mock_tts_synthesize_duration_tracks_text() -> None:
    provider = MockTTSProvider()
    result = await provider.synthesize("hello world")
    expected_ms = provider._estimate_ms("hello world")
    assert result.duration_ms == pytest.approx(expected_ms, abs=1.0)
    assert result.pcm


async def test_mock_tts_stream_yields_frames_and_concatenates() -> None:
    provider = MockTTSProvider()
    parts = []
    async for audio in provider.synthesize_stream(
        _iter_text(["This is a fairly long ", "sentence to speak out loud."])
    ):
        parts.append(audio)
    assert parts
    total = b"".join(a.pcm for a in parts)
    combined = AudioData(pcm=total, sample_rate=16_000)
    expected_ms = provider._estimate_ms("This is a fairly long sentence to speak out loud.")
    assert combined.duration_ms == pytest.approx(expected_ms, abs=200.0)
    assert len(parts) > 1


async def test_mock_tts_empty_input_yields_nothing() -> None:
    provider = MockTTSProvider()
    audio = [a async for a in provider.synthesize_stream(_iter_text([]))]
    assert audio == []


async def test_mock_tts_first_audio_latency_respected() -> None:
    provider = MockTTSProvider(first_audio_ms=80)
    started = asyncio.get_running_loop().time()
    async for _ in provider.synthesize_stream(_iter_text(["a reasonably long utterance here"])):
        elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000
        assert elapsed_ms >= 70
        break
