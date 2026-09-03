from __future__ import annotations

from sqlalchemy import JSON, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    turn_ordinal: Mapped[int] = mapped_column(Integer, default=0)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=0.0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
