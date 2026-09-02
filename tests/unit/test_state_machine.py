from __future__ import annotations

import pytest

from app.runtime.errors import StateViolationError
from app.runtime.state_machine import RuntimeState, StateMachine

ALL_STATES = list(RuntimeState)


def test_initial_state_is_idle() -> None:
    sm = StateMachine()
    assert sm.state is RuntimeState.IDLE


def test_valid_path_idle_to_listening() -> None:
    sm = StateMachine()
    sm.transition(RuntimeState.LISTENING)
    assert sm.state is RuntimeState.LISTENING


def test_full_conversation_cycle() -> None:
    sm = StateMachine()
    sm.transition(RuntimeState.LISTENING, reason="audio_connected")
    sm.transition(RuntimeState.PROCESSING, reason="user_turn:1")
    sm.transition(RuntimeState.THINKING, reason="tool_call")
    sm.transition(RuntimeState.PROCESSING, reason="tool_result")
    sm.transition(RuntimeState.SPEAKING, reason="tts_start")
    sm.transition(RuntimeState.LISTENING, reason="tts_done")
    assert sm.state is RuntimeState.LISTENING


def test_barge_in_cycle() -> None:
    sm = StateMachine()
    sm.transition(RuntimeState.LISTENING)
    sm.transition(RuntimeState.PROCESSING)
    sm.transition(RuntimeState.SPEAKING)
    sm.transition(RuntimeState.INTERRUPTED, reason="barge_in")
    sm.transition(RuntimeState.LISTENING, reason="speech_ended")
    assert sm.state is RuntimeState.LISTENING


def test_same_state_transition_rejected() -> None:
    sm = StateMachine(initial=RuntimeState.LISTENING)
    with pytest.raises(StateViolationError):
        sm.transition(RuntimeState.LISTENING)


def test_invalid_idle_to_speaking_rejected() -> None:
    sm = StateMachine()
    with pytest.raises(StateViolationError):
        sm.transition(RuntimeState.SPEAKING)
    assert sm.state is RuntimeState.IDLE


def test_invalid_listening_to_thinking_rejected() -> None:
    sm = StateMachine(initial=RuntimeState.LISTENING)
    with pytest.raises(StateViolationError):
        sm.transition(RuntimeState.THINKING)


def test_interrupted_to_processing_rejected() -> None:
    sm = StateMachine(initial=RuntimeState.INTERRUPTED)
    with pytest.raises(StateViolationError):
        sm.transition(RuntimeState.PROCESSING)


def test_closed_is_terminal() -> None:
    sm = StateMachine()
    sm.transition(RuntimeState.CLOSED)
    assert sm.state is RuntimeState.CLOSED
    for state in ALL_STATES:
        if state is not RuntimeState.CLOSED:
            with pytest.raises(StateViolationError):
                sm.transition(state)


def test_can_transition_query() -> None:
    sm = StateMachine()
    assert sm.can_transition(RuntimeState.LISTENING)
    assert not sm.can_transition(RuntimeState.SPEAKING)


def test_on_change_fires_with_reason() -> None:
    changes: list[tuple[str, str, str | None]] = []
    sm = StateMachine(on_change=lambda f, t, r: changes.append((f, t, r)))
    sm.transition(RuntimeState.LISTENING, reason="audio_connected")
    assert changes == [("idle", "listening", "audio_connected")]
    assert sm.last_reason == "audio_connected"


def test_validate_every_transition_is_reversible() -> None:
    for origin in RuntimeState:
        for target in RuntimeState:
            sm = StateMachine(initial=origin)
            if origin is not RuntimeState.CLOSED and target is not origin:
                if target is RuntimeState.CLOSED:
                    continue
                if sm.can_transition(target):
                    sm.transition(target)
                    assert sm.state is target
