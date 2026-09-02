from __future__ import annotations

from typing import Any


class VoxFlowError(Exception):
    """Base class for all VoxFlow runtime errors."""

    def __init__(self, message: str, *, code: str | None = None, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.context = context

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.context}


class ConfigurationError(VoxFlowError):
    pass


class ProviderError(VoxFlowError):
    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retryable: bool = False,
        status_code: int | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message, **context)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class ProviderUnavailableError(ProviderError):
    def __init__(self, provider: str, message: str = "provider not configured or unreachable", **context: Any) -> None:
        super().__init__(message, provider=provider, retryable=False, **context)


class ProviderTimeoutError(ProviderError):
    def __init__(self, provider: str, timeout_s: float, **context: Any) -> None:
        super().__init__(
            f"{provider} timed out after {timeout_s}s",
            provider=provider,
            retryable=True,
            timeout_s=timeout_s,
            **context,
        )


class SessionNotFoundError(VoxFlowError):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"session {session_id!r} not found",
            code="SESSION_NOT_FOUND",
            session_id=session_id,
        )


class SessionClosedError(VoxFlowError):
    def __init__(
        self,
        session_id: str,
        message: str = "session is closed or disconnected",
        **context: Any,
    ) -> None:
        super().__init__(message, code="SESSION_CLOSED", session_id=session_id, **context)


class StateViolationError(VoxFlowError):
    def __init__(
        self,
        from_state: str,
        to_state: str,
        *,
        session_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(
            f"invalid state transition {from_state!r} -> {to_state!r}" + (f" ({reason})" if reason else ""),
            code="STATE_VIOLATION",
            from_state=from_state,
            to_state=to_state,
            session_id=session_id,
        )


class StaleTurnError(VoxFlowError):
    def __init__(self, turn_id: int, current_turn_id: int, *, session_id: str | None = None) -> None:
        super().__init__(
            f"stale result from turn {turn_id} rejected (current turn is {current_turn_id})",
            code="STALE_TURN",
            turn_id=turn_id,
            current_turn_id=current_turn_id,
            session_id=session_id,
        )


class ToolError(VoxFlowError):
    def __init__(
        self,
        tool_name: str,
        message: str,
        *,
        retryable: bool = False,
        timeout_s: float | None = None,
        **context: Any,
    ) -> None:
        super().__init__(message, code="TOOL_ERROR", tool=tool_name, **context)
        self.tool_name = tool_name
        self.retryable = retryable
        self.timeout_s = timeout_s


class ToolNotFoundError(VoxFlowError):
    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"tool {tool_name!r} is not registered",
            code="TOOL_NOT_FOUND",
            tool=tool_name,
        )


class AudioProtocolError(VoxFlowError):
    pass


class EventBusFullError(VoxFlowError):
    def __init__(
        self,
        message: str = "event bus subscriber queue is full; event dropped",
        **context: Any,
    ) -> None:
        super().__init__(message, code="EVENT_BUS_FULL", **context)
