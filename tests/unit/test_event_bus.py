from __future__ import annotations

import asyncio

from app.runtime.event_bus import EventBus
from app.runtime.events import EventType, SpeechEnded, SpeechStarted, TranscriptPartial


def _handler_sink(store: list) -> None:
    def handler(event) -> None:
        store.append(event)

    return handler


async def test_subscriber_receives_matching_events() -> None:
    bus = EventBus(queue_maxsize=64)
    received: list = []
    sub = bus.subscribe(_handler_sink(received), event_types=frozenset({EventType.SPEECH_STARTED}))

    bus.publish(SpeechStarted(session_id="s1"))
    bus.publish(SpeechEnded(session_id="s1"))
    bus.publish(SpeechStarted(session_id="s1"))
    await sub.wait_idle()

    assert [e.type for e in received] == [EventType.SPEECH_STARTED, EventType.SPEECH_STARTED]
    await bus.close()


async def test_session_scoped_subscription() -> None:
    bus = EventBus(queue_maxsize=64)
    received_a: list = []
    sub_a = bus.subscribe(_handler_sink(received_a), session_id="sA")

    bus.publish(SpeechStarted(session_id="sA"))
    bus.publish(SpeechStarted(session_id="sB"))
    bus.publish(TranscriptPartial(session_id="sA", text="hi"))
    await sub_a.wait_idle()

    assert len(received_a) == 2
    await bus.close()


async def test_publish_preserves_order() -> None:
    bus = EventBus(queue_maxsize=256)
    received: list = []
    sub = bus.subscribe(_handler_sink(received), session_id="s1")

    for i in range(100):
        bus.publish(TranscriptPartial(session_id="s1", text=str(i)))
    await sub.wait_idle()

    assert [e.text for e in received] == [str(i) for i in range(100)]
    await bus.close()


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus(queue_maxsize=64)
    received: list = []
    sub = bus.subscribe(_handler_sink(received), session_id="s1")

    bus.publish(SpeechStarted(session_id="s1"))
    await sub.wait_idle()
    assert len(received) == 1

    sub.close()
    await asyncio.sleep(0.01)
    bus.publish(SpeechStarted(session_id="s1"))
    await asyncio.sleep(0.01)
    assert len(received) == 1
    await bus.close()


async def test_slow_consumer_drops_are_counted() -> None:
    bus = EventBus(queue_maxsize=2)
    blocker = asyncio.Event()

    async def slow(_event) -> None:
        await blocker.wait()

    sub = bus.subscribe(slow, session_id="s1")

    for _ in range(10):
        bus.publish(SpeechStarted(session_id="s1"))
    await asyncio.sleep(0.05)

    metrics = bus.metrics()
    assert metrics["published"] == 10
    assert metrics["dropped"] >= 1
    assert metrics["delivered"] + metrics["dropped"] <= 10

    sub.close()
    blocker.set()
    await bus.close()


async def test_publish_and_await_delivers_everything() -> None:
    bus = EventBus(queue_maxsize=64)
    received: list = []

    async def handler(event) -> None:
        await asyncio.sleep(0.005)
        received.append(event)

    bus.subscribe(handler, session_id="s1")

    for _ in range(5):
        await bus.publish_and_await(SpeechStarted(session_id="s1"))

    assert len(received) == 5
    await bus.close()


async def test_handler_error_does_not_kill_worker() -> None:
    bus = EventBus(queue_maxsize=64)
    received: list = []

    def flaky(event) -> None:
        if event.type is EventType.SPEECH_STARTED:
            raise ValueError("boom")
        received.append(event)

    sub = bus.subscribe(flaky, session_id="s1")
    bus.publish(SpeechStarted(session_id="s1"))
    bus.publish(SpeechEnded(session_id="s1"))
    await sub.wait_idle()

    assert bus.metrics()["handler_errors"] == 1
    assert len(received) == 1
    await bus.close()


async def test_close_stops_all_workers() -> None:
    bus = EventBus(queue_maxsize=64)
    subs = [bus.subscribe(_handler_sink([]), session_id="s1") for _ in range(5)]
    await bus.close()
    assert all(sub.worker.done() or sub.worker.cancelled() for sub in subs)
