# VoxFlow

Real-time Voice AI agent runtime: streaming audio, streaming STT/LLM/TTS, VAD,
turn detection, barge-in, cancellation, tool calling, latency observability and
an automated evaluation harness.

> Work in progress. Phase D in place: the streaming turn pipeline is wired end
> to end -- inbound PCM fans out to energy VAD and streaming STT, endpointing
> submits the user turn, and the LLM token stream is chunked at sentence
> boundaries into streaming TTS that pushes audio frames back over the audio
> WebSocket. Barge-in cancels the whole turn scope and stale audio cannot reach
> the browser after interruption. Architecture, setup and benchmarks are
> documented here as the runtime is built out.
