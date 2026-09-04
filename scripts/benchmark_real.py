"""Real-provider benchmark for the VoxFlow runtime.

Run a configurable number of full voice conversations (TTS-generated spoken
prompts -> streaming STT -> LLM -> tools -> streaming TTS) and record per-turn
latency plus a Mock-vs-Real and P50/P95 comparison.

Requires real provider keys: set PROVIDER_STT=deepgram, PROVIDER_LLM=openai and
PROVIDER_TTS=cartesia (or elevenlabs) plus the matching API keys/voice IDs in
your .env. Run with mock providers to (re)measure the baseline.

Usage:
    uv run python scripts/benchmark_real.py --turns 40
    uv run python scripts/benchmark_real.py --turns 40 --tts elevenlabs
    uv run python scripts/benchmark_real.py --tts mock          # baseline
"""

from __future__ import annotations

import argparse
import asyncio

from app.config import get_settings
from app.evaluation.bench import (
    bench_report_markdown,
    run_benchmark,
    run_mock_baseline,
    save_report,
    save_report_markdown,
)
from app.evaluation.support import eval_settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Real-provider VoxFlow benchmark")
    parser.add_argument("--turns", type=int, default=20, help="number of benchmark turns")
    parser.add_argument(
        "--tts",
        choices=["cartesia", "elevenlabs", "mock"],
        default=None,
        help="TTS provider (default: $BENCH_TTS or cartesia)",
    )
    parser.add_argument("--baseline", action="store_true", help="measure a fresh mock baseline and exit")
    parser.add_argument("--force-generate", action="store_true", help="regenerate cached prompt audio")
    args = parser.parse_args()

    env = get_settings()
    if args.tts is None:
        args.tts = getattr(env, "bench_tts", None) or "cartesia"
    if args.tts == "mock" or args.baseline:
        print("Measuring mock-provider baseline...")
        baseline = await run_mock_baseline(turns=min(args.turns, 12))
        print(f"  Mock baseline (median ms): {baseline}")
        return
    settings = eval_settings(
        provider_stt="deepgram",
        provider_llm="openai",
        provider_tts=args.tts,
        deepgram_api_key=env.deepgram_api_key,
        openai_api_key=env.openai_api_key,
        cartesia_api_key=env.cartesia_api_key,
        cartesia_voice_id=env.cartesia_voice_id,
        cartesia_model=env.cartesia_model,
        elevenlabs_api_key=env.elevenlabs_api_key,
        elevenlabs_voice_id=env.elevenlabs_voice_id,
        elevenlabs_model=env.elevenlabs_model,
        provider_stt_timeout_s=env.provider_stt_timeout_s,
        provider_llm_timeout_s=env.provider_llm_timeout_s,
        provider_tts_timeout_s=env.provider_tts_timeout_s,
    )

    print(f"Running {args.turns} real-provider turns (STT=deepgram, LLM=openai, TTS={args.tts})...")
    baseline = await run_mock_baseline()
    report = await run_benchmark(turns=args.turns, settings=settings, baseline_mock=baseline)
    json_path = save_report(report)
    md_path = save_report_markdown(report)
    print()
    print(bench_report_markdown(report))
    print(f"\nReports written:\n  JSON: {json_path}\n  Markdown: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
