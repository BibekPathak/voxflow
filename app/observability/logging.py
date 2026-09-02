from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.dev import ConsoleRenderer
from structlog.processors import JSONRenderer

from app.runtime.events import VoiceEvent

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_configured = False


def setup_logging(
    level: str = "INFO", *, json_output: bool | None = None, force: bool = False
) -> None:
    """Configure structlog once per process.

    Console output is the default for local development; JSON line output is used
    in production/container logs and matches the event log contract
    (one JSON object per line, structured fields).
    """
    global _configured
    if _configured and not force:
        return

    numeric = _LOG_LEVELS.get(level.upper(), logging.INFO)
    json_output = bool(json_output) if json_output is not None else False
    renderer: Any = JSONRenderer() if json_output else ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    logging.getLogger().setLevel(numeric)
    _configured = True


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name or "voxflow")


def log_event(
    event: VoiceEvent, message: str | None = None, *, level: str = "info", **extra: Any
) -> None:
    """Emit a structured log line for a runtime event.

    Every log line carries the correlation fields required for debugging
    "why was this conversation slow": session_id, conversation_id, turn_id,
    event_type, latency_ms and provider.
    """
    log = structlog.get_logger("voxflow.events")
    bound = log.bind(
        event_type=event.type.value,
        session_id=event.session_id,
        conversation_id=event.conversation_id,
        turn_id=event.turn_id,
        latency_ms=event.latency_ms,
        provider=event.provider,
        timestamp=event.timestamp,
    )
    if extra:
        bound = bound.bind(**extra)
    handler = getattr(bound, level, None) or bound.info
    handler(message or f"event {event.type.value}")
