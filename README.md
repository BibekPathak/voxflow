# VoxFlow

Real-time Voice AI agent runtime: streaming audio, streaming STT/LLM/TTS, VAD,
turn detection, barge-in, cancellation, tool calling, latency observability and
an automated evaluation harness.

> Work in progress. Phase C in place: provider abstraction layer with scripted
> offline mocks (streaming STT/LLM/TTS) and env-gated real adapters for Deepgram,
> OpenAI, Cartesia and ElevenLabs behind one factory. Architecture, setup and
> benchmarks are documented here as the runtime is built out.
