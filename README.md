# VoxFlow

Real-time Voice AI agent runtime: streaming audio, streaming STT/LLM/TTS, VAD,
turn detection, barge-in, cancellation, tool calling, latency observability and
an automated evaluation harness.

> Work in progress. Phase G in place: an automated evaluation harness running
> seven scripted voice-agent scenarios (simple question, tool call,
> interruption, backchannel, mid-sentence pause, tool failure, network
> degradation) against the real runtime with synthesized audio. Each scenario
> asserts voice-specific behavior and records TTFT/TTFA/E2E and interruption
> latencies; reports render as JSON or Markdown, with a compare view, and are
> exposed via POST/GET /evaluations. Architecture, setup and benchmarks are
> documented here as the runtime is built out.
