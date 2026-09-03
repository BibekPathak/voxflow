from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.memory.models import Base, TurnRow


def _build_engine(url: str) -> AsyncEngine:
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite") and ":memory:" in url:
        kwargs["poolclass"] = StaticPool
    return create_async_engine(url, **kwargs)


class ConversationStore:
    """Async persistence of conversation turns (PostgreSQL in production,
    SQLite for local dev/tests)."""

    def __init__(self, url: str) -> None:
        self._engine = _build_engine(url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def create_tables(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def add_message(
        self,
        *,
        conversation_id: str,
        session_id: str,
        turn_ordinal: int,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        row = TurnRow(
            conversation_id=conversation_id,
            session_id=session_id,
            turn_ordinal=turn_ordinal,
            role=role,
            content=content,
            created_at=time.time(),
            meta=meta or {},
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()

    async def recent(self, conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
        statement = (
            select(TurnRow).where(TurnRow.conversation_id == conversation_id).order_by(TurnRow.id.desc()).limit(limit)
        )
        async with self._session_factory() as session:
            result = await session.execute(statement)
            rows = result.scalars().all()
        rows = list(reversed(rows))
        return [
            {
                "id": row.id,
                "conversation_id": row.conversation_id,
                "session_id": row.session_id,
                "turn_ordinal": row.turn_ordinal,
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at,
                "meta": row.meta,
            }
            for row in rows
        ]

    async def close(self) -> None:
        await self._engine.dispose()


@dataclass(slots=True)
class HistoryEntry:
    """One short-term message retained for the current conversation context."""

    role: str
    content: str
    turn_ordinal: int = 0
    timestamp: float = field(default_factory=time.time)
