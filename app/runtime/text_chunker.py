from __future__ import annotations

_SENTENCE_END = frozenset({".", "!", "?"})
_CLAUSE_END = frozenset({";", ":", ","})
_SOFT_MIN_CHARS = 24
_HARD_MAX_CHARS = 160


class SentenceChunker:
    """Converts a token stream into TTS-ready text chunks.

    The LLM streams tokens that are often far shorter than a sentence. Sending
    each token to TTS would produce choppy, unnatural speech and hammer the TTS
    provider; waiting for the full response would destroy time-to-first-audio.
    This chunker buffers tokens and flushes at natural boundaries:

    * a sentence-final punctuation (``. ! ?``) once the buffer is a real
      sentence (>= 6 chars) -- the primary low-latency path;
    * clause punctuation (``; : ,``) once the buffer is long enough that TTS
      should already be speaking;
    * the full buffer when it hits a hard ceiling, so a long unpunctured run of
      text still reaches TTS without stalling.

    ``flush`` returns any remaining text at end-of-stream.
    """

    def __init__(self, *, soft_min_chars: int = _SOFT_MIN_CHARS, hard_max_chars: int = _HARD_MAX_CHARS) -> None:
        self.soft_min_chars = soft_min_chars
        self.hard_max_chars = hard_max_chars
        self._buffer = ""

    def push(self, token: str) -> list[str]:
        self._buffer += token
        chunks: list[str] = []
        while True:
            flushed = self._next_chunk()
            if flushed is None:
                break
            chunks.append(flushed)
        return chunks

    def _next_chunk(self) -> str | None:
        if not self._buffer:
            return None
        for index, char in enumerate(self._buffer):
            if char in _SENTENCE_END and index >= 5:
                return self._take(index + 1)
        if len(self._buffer) >= self.soft_min_chars:
            for index in range(len(self._buffer) - 1, -1, -1):
                if self._buffer[index] in _CLAUSE_END:
                    return self._take(index + 1)
        if len(self._buffer) >= self.hard_max_chars:
            return self._take(self.hard_max_chars)
        return None

    def flush(self) -> list[str]:
        if not self._buffer:
            return []
        chunk = self._take(len(self._buffer))
        return [chunk] if chunk else []

    @property
    def buffered(self) -> str:
        return self._buffer

    def _take(self, length: int) -> str:
        chunk = self._buffer[:length]
        self._buffer = self._buffer[length:]
        return chunk
