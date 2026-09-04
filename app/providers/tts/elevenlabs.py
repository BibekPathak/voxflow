from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from app.providers.meta import ProviderInfo
from app.providers.types import AudioData


class ElevenLabsTTSProvider:
    """Streaming TTS adapter for the ElevenLabs HTTP streaming API.

    Every text fragment triggers a streaming synthesis request for that fragment
    (reliable across accounts) and the returned raw PCM is forwarded chunk by
    chunk. For accounts that support it, ``output_format=pcm_16000`` yields
    16 kHz little-endian PCM directly; otherwise the content type is detected
    and non-PCM responses raise a clear error. Requires ``ELEVENLABS_API_KEY``
    and ``ELEVENLABS_VOICE_ID``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        voice_id: str,
        model_id: str = "eleven_turbo_v2_5",
        base_url: str = "https://api.elevenlabs.io",
        timeout_s: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def close(self) -> None:
        return None

    def metadata(self) -> ProviderInfo:
        return ProviderInfo(
            name="elevenlabs",
            kind="tts",
            vendor="ElevenLabs",
            model=self.model_id,
            streaming=True,
            voice_id=self.voice_id,
            endpoint=self.base_url,
        )

    async def synthesize_stream(
        self,
        text: AsyncIterator[str],
        *,
        sample_rate: int = 16_000,
    ) -> AsyncIterator[AudioData]:
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        body = {
            "text": "",
            "model_id": self.model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        timeout = httpx.Timeout(self.timeout_s, read=None)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async for piece in text:
                transcript = piece.strip()
                if not transcript:
                    continue
                body["text"] = transcript
                url = (
                    f"{self.base_url}/v1/text-to-speech/{self.voice_id}/stream"
                    f"?output_format=pcm_{sample_rate}&optimize_streaming_latency=3"
                )
                response = await client.post(url, headers=headers, json=body)
                if response.status_code != 200:
                    raise ConnectionError(
                        f"elevenlabs synthesis failed with status {response.status_code}: {response.text[:300]}"
                    )
                content_type = response.headers.get("content-type", "")
                if "pcm" not in content_type and "octet-stream" not in content_type and "audio" not in content_type:
                    raise ConnectionError(f"elevenlabs returned unexpected content-type {content_type!r}")
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield AudioData(pcm=chunk, sample_rate=sample_rate)

    async def synthesize(self, text: str, *, sample_rate: int = 16_000) -> AudioData:
        audio = bytearray()
        async for chunk in self.synthesize_stream(_one_shot(text), sample_rate=sample_rate):
            audio.extend(chunk.pcm)
        return AudioData(pcm=bytes(audio), sample_rate=sample_rate)


async def _one_shot(text: str) -> AsyncIterator[str]:
    yield text
