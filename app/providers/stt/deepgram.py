from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from urllib.parse import urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from app.providers.types import Transcript


class DeepgramSTTProvider:
    """Streaming STT adapter for Deepgram's WebSocket listen API.

    Sends raw 16-bit PCM as it arrives and relays interim partial results plus
    final transcripts (marked by Deepgram's ``speech_final`` / ``is_final``
    metadata). Audio frames are pushed from a concurrent producer task while the
    reader task forwards results, matching how the runtime feeds a live mic
    stream. Requires ``DEEPGRAM_API_KEY``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-3",
        endpoint: str = "wss://api.deepgram.com/v1/listen",
        timeout_s: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    async def close(self) -> None:
        return None

    def _listen_url(self, *, sample_rate: int, interim_results: bool, language: str | None) -> str:
        query = urlencode(
            {
                "model": self.model,
                "encoding": "linear16",
                "sample_rate": sample_rate,
                "channels": 1,
                "interim_results": "true" if interim_results else "false",
                "smart_format": "true",
                "punctuate": "true",
                **(language and {"language": language} or {}),
            }
        )
        parts = urlsplit(self.endpoint)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))

    async def transcribe_stream(
        self,
        audio: AsyncIterator[bytes],
        *,
        sample_rate: int = 16_000,
        interim_results: bool = True,
        language: str | None = None,
    ) -> AsyncIterator[Transcript]:
        url = self._listen_url(sample_rate=sample_rate, interim_results=interim_results, language=language)
        headers = {"Authorization": f"Token {self.api_key}"}
        try:
            async with connect(url, additional_headers=headers, open_timeout=self.timeout_s) as ws:

                async def producer() -> None:
                    try:
                        async for chunk in audio:
                            if chunk:
                                await ws.send(chunk)
                    finally:
                        try:
                            await ws.send(json.dumps({"type": "CloseStream"}))
                        except WebSocketException:
                            pass

                sender = asyncio.create_task(producer())
                try:
                    async for raw in ws:
                        message = json.loads(raw)
                        channel = (message.get("channel") or {}).get("alternatives") or []
                        if not channel:
                            continue
                        alternative = channel[0]
                        text = (alternative.get("transcript") or "").strip()
                        if not text:
                            continue
                        words = alternative.get("words")
                        is_final = bool(message.get("speech_final", False))
                        if is_final or interim_results:
                            yield Transcript(text=text, is_final=is_final, words=words, language=language)
                finally:
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
        except (TimeoutError, WebSocketException) as exc:
            raise ConnectionError(f"deepgram listen stream failed: {exc}") from exc
