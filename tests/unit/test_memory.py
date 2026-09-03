from __future__ import annotations

from app.memory.context import build_context_messages
from app.memory.conversation import ConversationStore, HistoryEntry
from app.providers.types import LLMMessage


def _entry(role: str, content: str, ordinal: int) -> HistoryEntry:
    return HistoryEntry(role=role, content=content, turn_ordinal=ordinal)


def test_context_window_keeps_recent_turns() -> None:
    history = [
        _entry("user", "first question", 1),
        _entry("assistant", "first answer", 1),
        _entry("user", "second question", 2),
        _entry("assistant", "second answer", 2),
    ]
    messages = build_context_messages(history, system_prompt="sys", user_text="third question", max_turns=2)
    contents = [(m.role, m.content) for m in messages]
    assert contents == [
        ("system", "sys"),
        ("user", "second question"),
        ("assistant", "second answer"),
        ("user", "third question"),
    ]


def test_context_drops_oldest_under_char_budget() -> None:
    long_old = "a" * 5000
    long_new = "b" * 5000
    history = [_entry("user", long_old, 1), _entry("assistant", long_new, 2)]
    messages = build_context_messages(history, system_prompt="sys", user_text="q", max_turns=10, max_chars=6000)
    roles = [m.role for m in messages]
    assert roles == ["system", "assistant", "user"]
    assert messages[1].content == long_new
    assert messages[-1].content == "q"


def test_context_always_includes_current_user() -> None:
    messages = build_context_messages(
        [_entry("user", "x" * 9000, 1)], system_prompt="sys", user_text="?", max_chars=500
    )
    assert messages[-1].role == "user"
    assert messages[-1].content == "?"


def test_message_types_are_valid() -> None:
    history = [_entry("user", "hello", 1), _entry("assistant", "hi", 1)]
    messages = build_context_messages(history, system_prompt="sys", user_text="how are you")
    assert all(isinstance(m, LLMMessage) for m in messages)


async def test_conversation_store_round_trip() -> None:
    store = ConversationStore("sqlite+aiosqlite:///:memory:")
    await store.create_tables()
    await store.add_message(
        conversation_id="c1",
        session_id="s1",
        turn_ordinal=1,
        role="user",
        content="my payment failed",
        meta={"lang": "en"},
    )
    await store.add_message(
        conversation_id="c1", session_id="s1", turn_ordinal=1, role="assistant", content="let me check"
    )
    rows = await store.recent("c1")
    assert [(r["role"], r["content"]) for r in rows] == [
        ("user", "my payment failed"),
        ("assistant", "let me check"),
    ]
    assert rows[0]["meta"] == {"lang": "en"}
    await store.close()


async def test_conversation_store_persists_across_reopen(tmp_path) -> None:
    path = tmp_path / "conv.db"
    url = f"sqlite+aiosqlite:///{path}"

    first = ConversationStore(url)
    await first.create_tables()
    await first.add_message(conversation_id="c1", session_id="s1", turn_ordinal=1, role="user", content="hello")
    await first.close()

    second = ConversationStore(url)
    await second.create_tables()
    await second.add_message(
        conversation_id="c1", session_id="s1", turn_ordinal=1, role="assistant", content="hi there"
    )
    rows = await second.recent("c1")
    assert len(rows) == 2
    assert rows[0]["content"] == "hello"
    assert rows[1]["content"] == "hi there"
    await second.close()
