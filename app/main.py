from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.audio import router as audio_router
from app.api.sessions import router as sessions_router
from app.config import Settings, get_settings
from app.memory.conversation import ConversationStore
from app.observability.logging import setup_logging
from app.runtime.errors import SessionNotFoundError, VoxFlowError
from app.runtime.session import SessionManager


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(level=settings.log_level, json_output=settings.json_logs)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store: ConversationStore | None = None
        if settings.persist_conversations:
            store = ConversationStore(settings.database_url)
            await store.create_tables()
        app.state.manager = SessionManager(settings, conversation_store=store)
        app.state.settings = settings
        yield
        await app.state.manager.close_all()
        if store is not None:
            await store.close()

    app = FastAPI(
        title="VoxFlow Voice AI Runtime",
        version=__version__,
        description="Real-time, event-driven Voice AI agent runtime.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(request: Request, exc: VoxFlowError) -> JSONResponse:
        return JSONResponse(status_code=404, content=exc.as_dict())

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name, "version": settings.app_version}

    app.include_router(sessions_router)
    app.include_router(audio_router)

    return app


app = create_app()
