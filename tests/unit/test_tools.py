from __future__ import annotations

import asyncio

import pytest

from app.runtime.errors import ToolNotFoundError
from app.tools import load_builtin_tools
from app.tools.registry import ToolError, ToolOutcome, ToolRegistry


def _make_tool(
    fn, *, name: str | None = None, timeout_s: float | None = None, max_retries: int | None = None
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(fn, name=name, timeout_s=timeout_s, max_retries=max_retries)
    return registry


async def test_schema_introspection_marks_required_and_defaults() -> None:
    async def do_thing(customer_id: str, limit: int = 5, note: str | None = None) -> dict:
        return {}

    registry = _make_tool(do_thing, name="do_thing")
    definition = registry.get("do_thing")
    schema = definition.parameters
    assert schema["required"] == ["customer_id"]
    props = schema["properties"]
    assert props["customer_id"]["type"] == "string"
    assert props["limit"]["type"] == "integer"
    assert props["note"]["type"] == "string"


async def test_execute_success_returns_structured_outcome() -> None:
    async def echo(value: str) -> dict:
        return {"value": value}

    outcome = await _make_tool(echo).execute("echo", '{"value": "hi"}')
    assert isinstance(outcome, ToolOutcome)
    assert outcome.ok is True
    assert outcome.value == {"value": "hi"}
    assert outcome.attempts == 1
    assert "hi" in outcome.content


async def test_unknown_tool_raises() -> None:
    with pytest.raises(ToolNotFoundError):
        await ToolRegistry().execute("nope", None)


async def test_timeout_is_retryable_and_eventually_fails() -> None:
    async def slow() -> None:
        await asyncio.sleep(0.2)

    registry = _make_tool(slow, max_retries=2, timeout_s=0.02)
    outcome = await registry.execute("slow", None, default_timeout_s=0.02, default_retries=0)
    assert outcome.ok is False
    assert outcome.attempts == 3
    assert "timed out" in (outcome.error or "")


async def test_retryable_tool_recovers_on_second_attempt() -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ToolError("flaky", "transient failure", retryable=True)
        return "recovered"

    registry = _make_tool(flaky, max_retries=2)
    outcome = await registry.execute("flaky", None)
    assert outcome.ok is True
    assert outcome.value == "recovered"
    assert outcome.attempts == 2


async def test_non_retryable_error_does_not_retry() -> None:
    calls = {"n": 0}

    async def fail() -> None:
        calls["n"] += 1
        raise ToolError("fail", "permanent")

    registry = _make_tool(fail, max_retries=3)
    outcome = await registry.execute("fail", None)
    assert outcome.ok is False
    assert calls["n"] == 1
    assert outcome.attempts == 1


async def test_invalid_json_arguments_return_failed_outcome() -> None:
    async def needs(value: str) -> str:
        return value

    outcome = await _make_tool(needs).execute("needs", "{not json")
    assert outcome.ok is False
    assert "invalid JSON" in (outcome.error or "")


async def test_plain_exception_becomes_failed_outcome() -> None:
    async def boom() -> None:
        raise ValueError("boom")

    outcome = await _make_tool(boom).execute("boom", None)
    assert outcome.ok is False
    assert "boom" in (outcome.error or "")


def test_builtin_tools_are_registered() -> None:
    registry = load_builtin_tools()
    assert set(registry.names()) >= {
        "search_customer",
        "get_recent_transactions",
        "inspect_payment",
        "create_support_ticket",
    }


async def test_builtin_tool_behaviors() -> None:
    registry = load_builtin_tools()
    customer = await registry.execute("search_customer", '{"email": "alice@example.com"}')
    assert customer.ok and customer.value["customer_id"] == "cust_1"

    payment = await registry.execute("inspect_payment", '{"payment_id": "pay_101"}')
    assert payment.ok and payment.value["decline_reason"] == "insufficient_funds"

    recent = await registry.execute("get_recent_transactions", '{"customer_id": "cust_1"}')
    assert recent.ok and recent.value["count"] == 2

    ticket = await registry.execute("create_support_ticket", '{"subject": "refund", "description": "wanted refund"}')
    assert ticket.ok and ticket.value["ticket_id"].startswith("tkt_")
    assert registry.get("inspect_payment").schema()["name"] == "inspect_payment"
