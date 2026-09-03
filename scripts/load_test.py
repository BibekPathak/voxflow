"""Concurrent load test for the VoxFlow runtime (mock providers).

Usage:
    uv run python scripts/load_test.py --concurrency 8 --turns 3
"""

from __future__ import annotations

import argparse
import asyncio

from app.evaluation.load import run_load_test


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a concurrent VoxFlow load test")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--turns", type=int, default=2)
    args = parser.parse_args()

    print(f"Running {args.concurrency} sessions x {args.turns} turns with mock providers...")
    report = await run_load_test(concurrency=args.concurrency, turns_per_session=args.turns)
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
