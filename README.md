# VoxFlow

Real-time Voice AI agent runtime: streaming audio, streaming STT/LLM/TTS, VAD,
turn detection, barge-in, cancellation, tool calling, latency observability and
an automated evaluation harness.

> Work in progress. Phase F in place: a per-session metrics collector that turns
> the event stream into a latency ledger -- TTFT, TTFA, end-to-end, transcript
> (first partial/final), interruption detection and TTS cancellation -- with
> median/P95 aggregation and counters, exposed via GET /sessions/{id}/metrics.
> Architecture, setup and benchmarks are documented here as the runtime is
> built out.
