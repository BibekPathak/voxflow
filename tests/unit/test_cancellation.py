from __future__ import annotations

import asyncio

import pytest

from app.runtime.cancellation import CancellationScope, ScopeClosedError, TurnCancelled


async def test_scope_spawn_and_complete() -> None:
    scope = CancellationScope(turn_id=1, name="turn-1")
    task = scope.spawn(asyncio.sleep(0.001, result="done"))
    assert await task == "done"
    assert not scope.cancelled
    assert scope.tasks == set()
    await scope.close()


async def test_scope_run_returns_value() -> None:
    scope = CancellationScope(turn_id=1, name="turn-1")
    result = await scope.run(asyncio.sleep(0, result=42))
    assert result == 42
    assert not scope.cancelled


async def test_cancel_cancels_all_tracked_tasks() -> None:
    scope = CancellationScope(turn_id=1, name="turn-1")
    scope.spawn(asyncio.sleep(10))
    scope.spawn(asyncio.sleep(10))
    scope.spawn(asyncio.sleep(10))
    assert len(scope.tasks) == 3

    await scope.cancel()

    assert scope.cancelled
    assert scope.cancelled_event.is_set()
    assert scope.tasks == set()


async def test_cancelled_tasks_raise_cancelled_error() -> None:
    scope = CancellationScope(turn_id=1, name="turn-1")

    async def work() -> None:
        await asyncio.sleep(10)

    task = scope.spawn(work())
    await scope.cancel()
    outcome = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(outcome[0], asyncio.CancelledError)


async def test_cancel_while_run_pending_raises_turn_cancelled() -> None:
    scope = CancellationScope(turn_id=2, name="turn-2")

    async def pipeline() -> None:
        await asyncio.sleep(10)

    awaiter = asyncio.create_task(scope.run(pipeline()))
    await asyncio.sleep(0)
    await scope.cancel()

    with pytest.raises(TurnCancelled):
        await awaiter


async def test_spawn_after_cancel_rejected() -> None:
    scope = CancellationScope(turn_id=1, name="turn-1")
    await scope.cancel()
    with pytest.raises(TurnCancelled):
        scope.spawn(lambda: asyncio.sleep(0))


async def test_spawn_after_close_rejected() -> None:
    scope = CancellationScope(turn_id=1, name="turn-1")
    await scope.close()
    with pytest.raises(ScopeClosedError):
        scope.spawn(lambda: asyncio.sleep(0))


async def test_wait_cancelled_releases() -> None:
    scope = CancellationScope(turn_id=1, name="turn-1")
    waiter = asyncio.create_task(scope.wait_cancelled())
    await asyncio.sleep(0)
    assert not waiter.done()
    await scope.cancel()
    await asyncio.wait_for(waiter, timeout=1)


async def test_join_collects_outcomes() -> None:
    scope = CancellationScope(turn_id=1, name="turn-1")
    scope.spawn(asyncio.sleep(0, result=1))
    scope.spawn(asyncio.sleep(0, result=2))
    outcomes = await scope.join()
    assert set(outcomes) == {1, 2}


async def test_cancel_idempotent() -> None:
    scope = CancellationScope(turn_id=1, name="turn-1")
    scope.spawn(asyncio.sleep(10))
    await scope.cancel()
    await scope.cancel()
    assert scope.cancelled


async def test_scope_isolates_turns() -> None:
    earlier = CancellationScope(turn_id=16, name="turn-16")
    current = CancellationScope(turn_id=17, name="turn-17")

    async def slow() -> str:
        await asyncio.sleep(10)
        return "stale"

    stale = earlier.spawn(slow())
    live = current.spawn(asyncio.sleep(0, result="fresh"))

    await earlier.cancel()
    stale_outcome = await asyncio.gather(stale, return_exceptions=True)
    assert isinstance(stale_outcome[0], asyncio.CancelledError)
    assert await live == "fresh"
    assert not current.cancelled
    await current.close()
