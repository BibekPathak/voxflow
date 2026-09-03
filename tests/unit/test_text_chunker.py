from __future__ import annotations

from app.runtime.text_chunker import SentenceChunker


def test_sentence_punctuation_flushes() -> None:
    chunker = SentenceChunker(soft_min_chars=24, hard_max_chars=160)
    chunks = chunker.push("Sure, I can help with that.")
    assert chunks == ["Sure, I can help with that."]
    assert chunker.buffered == ""


def test_multiple_sentences_flush_separately() -> None:
    chunker = SentenceChunker(soft_min_chars=24, hard_max_chars=160)
    chunks = chunker.push("First sentence. Second one. Third!")
    assert chunks == ["First sentence.", " Second one.", " Third!"]


def test_short_tokens_do_not_flush_early() -> None:
    chunker = SentenceChunker(soft_min_chars=24, hard_max_chars=160)
    assert chunker.push("Hello ") == []
    assert chunker.push("there ") == []
    assert chunker.push("friend.") == ["Hello there friend."]


def test_no_punctuation_flushes_at_hard_cap() -> None:
    chunker = SentenceChunker(soft_min_chars=24, hard_max_chars=40)
    text = "a" * 39 + "b" * 10
    chunks = chunker.push(text)
    assert chunks == [text[:40]]
    assert chunker.buffered == "b" * 9


def test_long_clause_flushes_at_comma() -> None:
    chunker = SentenceChunker(soft_min_chars=20, hard_max_chars=160)
    chunks = chunker.push("this is a fairly long clause without end punctuation, and then it continues")
    assert chunks == ["this is a fairly long clause without end punctuation,"]
    assert "continues" in chunker.buffered


def test_flush_returns_remainder() -> None:
    chunker = SentenceChunker(soft_min_chars=24, hard_max_chars=160)
    chunker.push("no sentence boundary yet here")
    assert chunker.buffered == "no sentence boundary yet here"
    assert chunker.flush() == ["no sentence boundary yet here"]
    assert chunker.buffered == ""


def test_empty_flush_is_noop() -> None:
    chunker = SentenceChunker()
    assert chunker.flush() == []
