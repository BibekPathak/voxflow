from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.config import Settings, get_settings
from app.observability.logging import setup_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(level=settings.log_level, json_output=settings.json_logs)

    app = FastAPI(
        title="VoxFlow Voice AI Runtime",
        version=__version__,
        description="Real-time, event-driven Voice AI agent runtime.",
    )

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name, "version": settings.app_version}

    return app


app = create_app()
