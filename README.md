# VoxFlow

Real-time Voice AI agent runtime: streaming audio, streaming STT/LLM/TTS, VAD,
turn detection, barge-in, cancellation, tool calling, latency observability and
an automated evaluation harness.

> Work in progress. Phase B in place: explicit runtime state machine, session
> manager + WebSocket audio/events transport, energy VAD, jitter/recording audio
> buffers, and a conservative turn/endpoint detector. Architecture, setup and
> benchmarks are documented here as the runtime is built out.
