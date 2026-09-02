from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any


class TurnCancelled(Exception):
    """Raised inside a unit of work when its cancellation scope is cancelled."""


class ScopeClosedError(RuntimeError):
    pass


class CancellationScope:
    """Groups the asynchronous work belonging to one unit of work (typically one
    turn) so the whole group can be cancelled together and awaited to completion.

    A scope is single-use: once cancelled or closed it rejects new work. Tasks
    registered through :meth:`spawn` run concurrently; :meth:`cancel` requests
    cancellation of every tracked task and waits for them to unwind. This is the
    primitive the orchestrator uses to make barge-in / interruption a first-class
    operation rather than best-effort task killing.
    """

    def __init__(self, turn_id: int | None = None, *, name: str | None = None) -> None:
        self.turn_id = turn_id
        self.name = name
        self._tasks: set[asyncio.Task[Any]] = set()
        self._cancelled = False
        self._closed = False
        self._cancel_event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def cancelled_event(self) -> asyncio.Event:
        return self._cancel_event

    @property
    def tasks(self) -> set[asyncio.Task[Any]]:
        return set(self._tasks)

    def spawn(
        self,
        work: Coroutine[Any, Any, Any] | Callable[[], Coroutine[Any, Any, Any]],
        *,
        name: str | None = None,
    ) -> asyncio.Task[Any]:
        if self._cancelled:
            raise TurnCancelled(f"scope {self.name!r} was cancelled")
        if self._closed:
            raise ScopeClosedError(f"scope {self.name!r} is closed; refusing new work")
        coro = work() if callable(work) else work
        task_name = name or f"{self.name or 'scope'}.task"
        task = asyncio.create_task(coro, name=task_name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def run(self, coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> Any:
        """Spawn a task, await it, and translate a cancelled scope into a
        :class:`TurnCancelled` so callers can clean up deterministically."""
        task = self.spawn(coro, name=name)
        try:
            return await task
        except asyncio.CancelledError:
            await self.cancel()
            raise TurnCancelled(f"scope {self.name!r} cancelled while awaiting work") from None

    async def cancel(self) -> None:
        """Cancel every tracked task and wait for them all to finish unwinding."""
        if self._cancelled:
            return
        self._cancelled = True
        self._closed = True
        self._cancel_event.set()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def close(self) -> None:
        """Gracefully close the scope: cancel outstanding work if any remains."""
        if self._closed:
            return
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def wait_cancelled(self) -> None:
        await self._cancel_event.wait()

    async def join(self) -> list[Any]:
        """Await all tracked tasks and return their outcomes (or exceptions)."""
        tasks = list(self._tasks)
        if not tasks:
            return []
        return await asyncio.gather(*tasks, return_exceptions=True)


async def wait_for_with_timeout(
    awaitable: Awaitable[Any], timeout_s: float, *, message: str = "operation timed out"
) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_s)
    except TimeoutError as exc:
        raise TimeoutError(message) from exc
