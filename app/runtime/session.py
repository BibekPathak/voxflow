from __future__ import annotations

import uuid
from typing import Any

from app.config import Settings
from app.memory.conversation import ConversationStore
from app.runtime.errors import SessionNotFoundError
from app.runtime.orchestrator import SessionRuntime


class SessionManager:
    """Registry and factory for live session runtimes.

    Sessions live in process memory keyed by ``session_id``. Each runtime owns
    its event bus, state machine, audio gateway, providers, tools and
    cancellation scopes; the manager is deliberately dumb (create/get/list/
    close) so the runtime can be swapped for a redis-backed registry without
    touching the API layer.
    """

    def __init__(self, settings: Settings, conversation_store: ConversationStore | None = None) -> None:
        self.settings = settings
        self._conversation_store = conversation_store
        self._sessions: dict[str, SessionRuntime] = {}

    def create_session(self) -> SessionRuntime:
        session_id = uuid.uuid4().hex
        conversation_id = uuid.uuid4().hex
        runtime = SessionRuntime(
            session_id=session_id,
            conversation_id=conversation_id,
            settings=self.settings,
            conversation_store=self._conversation_store,
        )
        self._sessions[session_id] = runtime
        return runtime

    def get(self, session_id: str) -> SessionRuntime:
        runtime = self._sessions.get(session_id)
        if runtime is None:
            raise SessionNotFoundError(session_id)
        return runtime

    def get_or_none(self, session_id: str) -> SessionRuntime | None:
        return self._sessions.get(session_id)

    def list(self) -> list[SessionRuntime]:
        return list(self._sessions.values())

    def snapshot(self, runtime: SessionRuntime) -> dict[str, Any]:
        return runtime.snapshot()

    async def close_all(self) -> None:
        for runtime in self._sessions.values():
            await runtime.dispose("manager_shutdown")
        self._sessions.clear()
