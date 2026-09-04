from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from app.providers.meta import ProviderInfo
from app.providers.types import AudioData


class CartesiaTTSProvider:
    """Streaming TTS adapter for Cartesia's context-based WebSocket API.

    Text fragments are sent as ``continue`` messages on a single connection and
    the audio chunks returned by the server are forwarded as raw PCM. Requires
    ``CARTESIA_API_KEY`` and ``CARTESIA_VOICE_ID``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        voice_id: str,
        model_id: str = "sonic-2",
        url: str = "wss://api.cartesia.ai/tts/websocket",
        sample_rate: int = 16_000,
        timeout_s: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.url = url
        self.sample_rate = sample_rate
        self.timeout_s = timeout_s

    async def close(self) -> None:
        return None

    def metadata(self) -> ProviderInfo:
        return ProviderInfo(
            name="cartesia",
            kind="tts",
            vendor="Cartesia",
            model=self.model_id,
            streaming=True,
            voice_id=self.voice_id,
            endpoint=self.url,
        )

    def _connect_url(self) -> str:
        parts = urlsplit(self.url)
        query = urlencode({"api_version": "2024-06-10", "model_id": self.model_id})
        return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))

    def _output_format(self) -> dict[str, Any]:
        return {"container": "raw", "encoding": "pcm_s16le", "sample_rate": self.sample_rate}

    def _payload(self, transcript: str, *, first: bool, context_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "tts",
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": self._output_format(),
            "language": "en",
            "transcript": transcript,
            "continue": not first,
        }
        if not first:
            payload["context_id"] = context_id
        return payload

    async def synthesize_stream(
        self,
        text: AsyncIterator[str],
        *,
        sample_rate: int = 16_000,
    ) -> AsyncIterator[AudioData]:
        url = self._connect_url()
        headers = {"X-API-Key": self.api_key}
        context_id: str | None = None
        first = True
        try:
            async with connect(url, additional_headers=headers, open_timeout=self.timeout_s) as ws:

                async def producer() -> None:
                    nonlocal context_id, first
                    async for piece in text:
                        transcript = piece.strip()
                        if not transcript:
                            continue
                        await ws.send(json.dumps(self._payload(transcript, first=first, context_id=context_id)))
                        first = False

                sender = asyncio.create_task(producer())
                try:
                    while True:
                        raw = await ws.recv()
                        message = json.loads(raw)
                        msg_type = message.get("type")
                        if msg_type == "chunk":
                            data = message.get("data")
                            if data:
                                pcm = base64.b64decode(data)
                                yield AudioData(pcm=pcm, sample_rate=sample_rate or self.sample_rate)
                        elif msg_type == "context":
                            context_id = context_id or message.get("context_id")
                        elif msg_type == "done":
                            break
                finally:
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
        except (TimeoutError, WebSocketException, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise ConnectionError(f"cartesia tts stream failed: {exc}") from exc

    async def synthesize(self, text: str, *, sample_rate: int = 16_000) -> AudioData:
        audio = bytearray()
        async for chunk in self.synthesize_stream(_one_shot(text), sample_rate=sample_rate):
            audio.extend(chunk.pcm)
        return AudioData(pcm=bytes(audio), sample_rate=sample_rate)


async def _one_shot(text: str) -> AsyncIterator[str]:
    yield text
