from __future__ import annotations

from app.evaluation.load import run_load_test


async def test_concurrent_sessions_complete_and_isolated() -> None:
    result = await run_load_test(concurrency=4, turns_per_session=2)
    assert result["expected_turns"] == 8
    assert result["turns_completed"] == result["expected_turns"]
    assert result["sessions_completed"] == 4
    assert result["errors"] == 0
    assert result["stale_turns"] == 0
    assert result["elapsed_ms"] > 0


async def test_concurrent_load_is_fast_enough() -> None:
    result = await run_load_test(concurrency=6, turns_per_session=2)
    assert result["turns_completed"] == 12
    assert result["errors"] == 0
    # 12 turns in well under 60s keeps the guard generous to avoid CI flakiness.
    assert result["elapsed_ms"] < 60_000
