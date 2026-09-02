from __future__ import annotations

import asyncio
import inspect
import itertools
from collections.abc import Callable
from typing import Any

from app.runtime.events import EventType, VoiceEvent

EventHandler = Callable[[VoiceEvent], Any]


class _Close:
    pass


_CLOSE = _Close()


class EventSubscription:
    """A named, closable subscription bound to a FIFO delivery queue.

    Ordering guarantee: events published to a given subscription are delivered to
    its handler one at a time, in publication order, because a single background
    worker consumes the queue sequentially. This is what lets downstream stages
    (VAD, dispatcher) rely on deterministic ordering without locks.
    """

    __slots__ = ("id", "bus", "session_id", "event_types", "handler", "queue", "worker", "closed")

    def __init__(
        self,
        id: int,
        bus: EventBus,
        handler: EventHandler,
        *,
        session_id: str | None,
        event_types: frozenset[EventType] | None,
    ) -> None:
        self.id = id
        self.bus = bus
        self.handler = handler
        self.session_id = session_id
        self.event_types = event_types
        self.queue: asyncio.Queue[VoiceEvent | _Close] = asyncio.Queue(maxsize=bus.queue_maxsize)
        self.closed = False
        self.worker: asyncio.Task[None] | None = None

    @property
    def active(self) -> bool:
        return not self.closed

    def _start(self) -> None:
        self.worker = asyncio.create_task(self._run(), name=f"event-subscriber-{self.id}")

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            if isinstance(item, _Close):
                self.queue.task_done()
                return
            try:
                result = self.handler(item)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                self.bus._handle_handler_error(self, item)
            finally:
                self.queue.task_done()

    async def wait_idle(self) -> None:
        """Block until every already-published event has been handled."""
        await self.queue.join()

    def close(self) -> None:
        self.bus.unsubscribe(self)


class EventBus:
    """Process-local publish/subscribe bus with per-subscriber FIFO queues.

    ``publish`` fans an event out to every matching subscription without ever
    blocking the publisher: events are appended to bounded queues with
    ``put_nowait``. A saturated queue (a slow consumer) causes events to be
    dropped and counted rather than stalling the realtime audio loop.

    Subscriptions may filter by ``session_id`` and/or event type. Session-scoped
    runtime components subscribe to their own session; the dashboard/event
    stream subscribes to a session for observability.
    """

    def __init__(self, *, name: str = "events", queue_maxsize: int = 1024) -> None:
        self.name = name
        self.queue_maxsize = queue_maxsize
        self._subscriptions: dict[int, EventSubscription] = {}
        self._ids = itertools.count(1)
        self._published = 0
        self._delivered = 0
        self._dropped = 0
        self._handler_errors = 0
        self._closing = False

    def subscribe(
        self,
        handler: EventHandler,
        *,
        session_id: str | None = None,
        event_types: frozenset[EventType] | None = None,
    ) -> EventSubscription:
        if self._closing:
            raise RuntimeError(f"bus {self.name!r} is closed; cannot subscribe")
        sub = EventSubscription(
            next(self._ids),
            self,
            handler,
            session_id=session_id,
            event_types=event_types,
        )
        self._subscriptions[sub.id] = sub
        sub._start()
        return sub

    def unsubscribe(self, sub: EventSubscription) -> None:
        if sub.closed:
            return
        sub.closed = True
        self._subscriptions.pop(sub.id, None)
        if sub.worker is not None:
            sub.worker.cancel()

    def _matches(self, sub: EventSubscription, event: VoiceEvent) -> bool:
        if sub.session_id is not None and sub.session_id != event.session_id:
            return False
        if sub.event_types is not None and event.type not in sub.event_types:
            return False
        return True

    def publish(self, event: VoiceEvent) -> dict[str, int]:
        """Deliver an event to all matching subscriptions (non-blocking).

        Returns delivery counters. Raises nothing; saturation is handled by
        dropping and accounting.
        """
        self._published += 1
        delivered = 0
        dropped = 0
        for sub in list(self._subscriptions.values()):
            if not sub.active or not self._matches(sub, event):
                continue
            try:
                sub.queue.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                dropped += 1
        self._delivered += delivered
        self._dropped += dropped
        return {"published": 1, "delivered": delivered, "dropped": dropped}

    async def publish_and_await(self, event: VoiceEvent) -> dict[str, int]:
        counts = self.publish(event)
        subscribers = [s for s in self._subscriptions.values() if self._matches(s, event)]
        if subscribers:
            await asyncio.gather(*(s.wait_idle() for s in subscribers))
        return counts

    def _handle_handler_error(self, sub: EventSubscription, event: VoiceEvent) -> None:
        self._handler_errors += 1

    def metrics(self) -> dict[str, int]:
        return {
            "published": self._published,
            "delivered": self._delivered,
            "dropped": self._dropped,
            "handler_errors": self._handler_errors,
            "subscribers": len(self._subscriptions),
        }

    async def close(self) -> None:
        self._closing = True
        workers: list[asyncio.Task[None]] = []
        for sub in list(self._subscriptions.values()):
            sub.closed = True
            workers.append(sub.worker)
            if sub.worker is not None:
                sub.worker.cancel()
        self._subscriptions.clear()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
