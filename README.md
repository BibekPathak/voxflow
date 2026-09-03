# VoxFlow

Real-time Voice AI agent runtime: streaming audio, streaming STT/LLM/TTS, VAD,
turn detection, barge-in, cancellation, tool calling, latency observability and
an automated evaluation harness.

> Work in progress. Phase E in place: a tool registry with signature-derived
> schemas, per-tool timeout/retry policies and structured outcomes, four
> builtin support tools (search_customer, get_recent_transactions,
> inspect_payment, create_support_ticket), a sliding-window conversation
> context manager, and async turn persistence to PostgreSQL/SQLite. The
> streaming pipeline now loops LLM passes through tool execution before
> speaking its answer. Architecture, setup and benchmarks are documented here
> as the runtime is built out.
