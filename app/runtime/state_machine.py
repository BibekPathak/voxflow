from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from app.runtime.errors import StateViolationError

StateChangeHandler = Callable[[str, str, str | None], None]


class RuntimeState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    CLOSED = "closed"


_IDLE: set[RuntimeState] = {RuntimeState.LISTENING, RuntimeState.CLOSED}
_LISTENING: set[RuntimeState] = {RuntimeState.PROCESSING, RuntimeState.CLOSED}
_PROCESSING: set[RuntimeState] = {
    RuntimeState.THINKING,
    RuntimeState.SPEAKING,
    RuntimeState.INTERRUPTED,
    RuntimeState.LISTENING,
    RuntimeState.CLOSED,
}
_THINKING: set[RuntimeState] = {
    RuntimeState.PROCESSING,
    RuntimeState.SPEAKING,
    RuntimeState.INTERRUPTED,
    RuntimeState.LISTENING,
    RuntimeState.CLOSED,
}
_SPEAKING: set[RuntimeState] = {
    RuntimeState.LISTENING,
    RuntimeState.INTERRUPTED,
    RuntimeState.CLOSED,
}
_INTERRUPTED: set[RuntimeState] = {RuntimeState.LISTENING, RuntimeState.CLOSED}

_ALLOWED_TRANSITIONS: dict[RuntimeState, frozenset[RuntimeState]] = {
    RuntimeState.IDLE: frozenset(_IDLE),
    RuntimeState.LISTENING: frozenset(_LISTENING),
    RuntimeState.PROCESSING: frozenset(_PROCESSING),
    RuntimeState.THINKING: frozenset(_THINKING),
    RuntimeState.SPEAKING: frozenset(_SPEAKING),
    RuntimeState.INTERRUPTED: frozenset(_INTERRUPTED),
    RuntimeState.CLOSED: frozenset(),
}


class StateMachine:
    """Explicit conversation/session state machine.

    Every transition is validated against the allowed-transition table and a
    rejected transition raises :class:`StateViolationError` rather than silently
    corrupting session semantics. This is the backstop that prevents, for
    example, starting TTS in a disconnected session or resuming a response from a
    cancelled turn.
    """

    def __init__(
        self,
        initial: RuntimeState = RuntimeState.IDLE,
        *,
        on_change: StateChangeHandler | None = None,
    ) -> None:
        self._state = initial
        self._on_change = on_change
        self._last_reason: str | None = None

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def last_reason(self) -> str | None:
        return self._last_reason

    def can_transition(self, to_state: RuntimeState) -> bool:
        if to_state == self._state:
            return False
        return to_state in _ALLOWED_TRANSITIONS[self._state]

    def transition(self, to_state: RuntimeState, *, reason: str | None = None) -> None:
        if to_state == self._state:
            raise StateViolationError(self._state.value, to_state.value, reason=reason)
        allowed = _ALLOWED_TRANSITIONS[self._state]
        if to_state not in allowed:
            raise StateViolationError(self._state.value, to_state.value, reason=reason)
        from_state = self._state
        self._state = to_state
        self._last_reason = reason
        if self._on_change is not None:
            self._on_change(from_state.value, to_state.value, reason)

    def describe_transitions(self) -> dict[str, list[str]]:
        return {state.value: sorted(to.value for to in allowed) for state, allowed in _ALLOWED_TRANSITIONS.items()}
