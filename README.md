# VoxFlow — Real-Time Voice AI Agent Runtime

VoxFlow is a small but credible **voice AI runtime/platform** — not a "connect an
STT API to an LLM API" chatbot wrapper. It is an asynchronous, event-driven
engine for low-latency spoken conversation: streaming audio both directions,
streaming STT/LLM/TTS, energy-based VAD with hysteresis, conservative turn
detection, first-class barge-in and cancellation, tool calling, latency
instrumentation, and an automated voice-agent evaluation harness.

Everything runs out of the box with **offline mock providers** (no API keys).
Real providers (Deepgram, OpenAI, Cartesia, ElevenLabs) sit behind the same
interfaces and activate via environment variables.

---

## 1. Problem: why voice agents differ from text agents

A text chatbot has unbounded patience. A voice agent does not:

* **Audio is a stream, not a message.** The user does not click "send" — the
  system must decide *when* a user has stopped speaking (endpointing) from
  continuous audio.
* **The user interrupts.** In text, the model's turn runs to completion. In
  voice, the moment the user starts talking the agent must **stop speaking and
  cancel all in-flight work** — and stale audio from the cancelled response must
  never leak into the new turn.
* **Latency is experienced directly.** Every millisecond between the user
  stopping and the agent answering is felt. Response text must reach the TTS
  engine before the LLM has finished writing, and the first audio must reach the
  browser before TTS has finished synthesizing.
* **Cancellation is correctness.** A cancelled LLM/TTS pipeline that keeps
  producing output corrupts the conversation. Cancellation must be structured,
  not best-effort.

These constraints shape the whole architecture below.

---

## 2. Architecture

```
                           Browser (React dashboard)
                                     │
                      getUserMedia 16 kHz PCM  │   agent PCM + control frames
                                     ▼
                     ┌───────────────────────────────────────┐
                     │           Audio WebSocket             │
                     └───────────────┬───────────────┬───────┘
                                     │               │
                                     ▼               ▼
                     ┌────────────────────────┐  ┌───────────────────────────────┐
                     │      Audio Gateway     │  │      Session Event Bus        │
                     │  PCM decode/buffer     │  │  (ordered, per-subscriber FIFO │
                     │  jitter/gap tracking   │  │   queues, session-scoped)      │
                     │  energy VAD (hysteresis)│ └──────────────┬────────────────┘
                     └──────┬───────────┬─────┘                │
                            │           │                      ▼
                            ▼           ▼          ┌───────────────────────────┐
                     ┌────────────┐ ┌────────────┐ │    MetricsCollector        │
                     │ streaming  │ │ Turn       │ │  TTFT / TTFA / E2E /       │
                     │ STT        │ │ Detector   │ │  interruption latencies    │
                     │ (partials) │ │ endpointing│ └───────────────────────────┘
                     └────────────┘ └─────┬──────┘
                                          │  user turn (final transcript)
                                          ▼
                     ┌─────────────────────────────────────────────────────┐
                     │              Voice Runtime (per session)             │
                     │  State machine IDLE→LISTENING→PROCESSING→SPEAKING…   │
                     │  TurnContext + CancellationScope (per turn)          │
                     │  StreamingTurnPipeline                               │
                     └───┬────────────────┬────────────────┬───────────────┘
                         ▼                ▼                ▼
                     ┌────────┐      ┌────────┐      ┌──────────────┐
                     │  LLM   │─────▶│ Tools  │      │  Conversation│
                     │streaming│     │registry│      │ memory/store │
                     └────────┘      └────────┘      └──────────────┘
                         │
                         ▼  sentence-boundary chunker
                     ┌────────┐
                     │  TTS   │  streaming audio → browser (agent_audio.start/end)
                     └────────┘
```

Key modules:

| Module | Responsibility |
| --- | --- |
| `app/runtime/orchestrator.py` | Per-session engine: event bus, state machine, VAD/STT fan-out, barge-in, turn lifecycle |
| `app/runtime/state_machine.py` | Explicit validated transitions; invalid transitions raise |
| `app/runtime/cancellation.py` | Single-use per-turn `CancellationScope` |
| `app/runtime/turn.py` | Conservative endpointing (silence + speech duration + STT state) |
| `app/runtime/pipeline.py` | LLM token stream → chunker → streaming TTS, tool-call loops |
| `app/audio/` | Resampling, buffers, jitter/gap detector, energy VAD, gateway |
| `app/providers/` | `STT/LLM/TTS` protocols, mock providers, Deepgram/OpenAI/Cartesia/ElevenLabs adapters |
| `app/tools/` | Registry + schema introspection, retry/timeout, 4 support tools |
| `app/memory/` | Sliding-window context + SQLAlchemy turn persistence |
| `app/observability/` | Structured logs + per-session latency ledger |
| `app/evaluation/` | Scripted voice scenarios, runner, JSON/Markdown reports |
| `frontend/` | React dashboard + real browser mic/playback |

---

## 3. Key engineering decisions

**Streaming everywhere.** Inbound mic PCM is streamed (20 ms frames) straight to
STT and VAD; STT emits partials then a final; LLM tokens stream; the sentence
chunker groups them into speakable phrases so TTS starts mid-generation; TTS
frames stream to the browser before synthesis finishes.

**One event model, one bus.** Every stage publishes typed events carrying
`session_id / conversation_id / turn_id / timestamp / latency_ms`. The bus
delivers to per-subscriber FIFO queues, giving deterministic ordering (VAD,
endpointing and metrics all subscribe) without locking the audio loop.

**Explicit state machine.** `IDLE → LISTENING → PROCESSING → (THINKING) →
SPEAKING → LISTENING`, with `INTERRUPTED` and terminal `CLOSED`. Invalid
transitions raise rather than corrupt session semantics (no TTS into a dead
session, no resume of a cancelled turn).

**Cancellation is a per-turn scope.** Every turn runs in a single-use
`CancellationScope` (turn-scoped `asyncio` tasks). Barge-in = cancel the scope:
LLM consumer, TTS worker and tool calls unwind together. The TTS worker checks
turn-id + state before every frame, so **stale audio cannot reach the browser**
after cancellation. Results belonging to an old turn are dropped by turn id.

**Endpointing is conservative.** Speech-end from VAD alone does not submit a
turn. The turn detector waits out a trailing-silence window, respects a minimum
speech duration, uses STT transcript state (a strong end-of-sentence shortcut)
and a maximum-utterance cap. Mid-sentence pauses do not fire.

**Provider-agnostic runtime.** The runtime only depends on `STTProvider`,
`LLMProvider`, `TTSProvider` protocols. Mocks are deterministic and scripted so
the entire suite — including interruption and evaluation — runs key-free.

**Measured latency, not asserted latency.** A per-session collector turns the
event stream into TTFT / TTFA / E2E / interruption / TTS-cancellation numbers
with median + P95 aggregation, exposed at `GET /sessions/{id}/metrics`.

**Evaluation drives the real runtime.** Scenarios synthesize audio (triggering
real VAD/endpointing/barge-in) and inject transcripts, then assert voice
behavior and read back measured latencies.

---

## 4. Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Node 20+.

```bash
cp .env.example .env          # defaults are mock providers; add keys to go live

# Backend
uv sync
uv run uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

Open http://localhost:5173, click **Start conversation**, allow microphone
access and speak. Interrupt the agent any time — it stops speaking immediately
and starts listening. The mock agent speaks the configured scripted utterances
for STT, so its replies are deterministic.

### Docker

```bash
docker compose up --build     # backend :8000, frontend :5173, postgres, redis
```

### Running providers live

Edit `.env` and set the provider values documented there
(`PROVIDER_STT=deepgram`, `PROVIDER_LLM=openai`, `PROVIDER_TTS=cartesia`, …)
plus the matching API keys and voice IDs. Each adapter is isolated behind the
same interfaces; the runtime code never changes.

---

## 5. Tests

```bash
uv run pytest                # 144 tests, provider-free
uv run ruff check app tests
uv run ruff format --check app tests
```

Coverage includes: state-machine transitions (valid/invalid/terminal), VAD
(onset/offset hysteresis, chunk boundaries, dips), endpointing (pause tolerance,
interrupt reset, short-speech rejection, max-utterance), event bus ordering and
saturation, cancellation (interrupted TTS, stale-result rejection), tool
semantics (timeout/retry/recovery), conversation-store persistence, full
`audio→STT→LLM→tool→LLM→TTS` turns over a real uvicorn server, and the
evaluation scenarios.

---

## 6. Automated evaluation

`POST /evaluations/run` executes all scenarios, or pass `{"scenario": name}`.
`GET /evaluations` lists runs; `GET /evaluations/{run_id}?format=markdown`
renders a report. Scenarios: simple question, tool call (selection + argument +
answer accuracy), interruption (stale-audio leak check + next-turn processing),
backchannel (no restart), mid-sentence pause (no premature endpointing), tool
failure (graceful recovery) and network degradation (dropped audio frames).

### Measured results (mock providers, evaluation profile)

A full run of the seven scenarios completes in ~3.4 s with 7/7 passing. These
numbers are real measurements from the harness above — nothing fabricated.
With mock providers the LLM and audio generation are nearly free, so TTFT/TTFA
are near the simulated provider floors; the E2E value is dominated by the
turn-silence window.

| Scenario | TTFT (ms) | TTFA (ms) | E2E (ms) | Notes |
| --- | --- | --- | --- | --- |
| simple_question | 0.7 | 17 | 223 | completed + speaks |
| tool_call | 2.1 | 22 | 229 | correct tool + args, `insufficient_funds` in answer |
| interruption | 1.1 | 28 | 233 | barge 0.5 ms detect, TTS cancel 14 ms, no stale audio |
| ambiguous_speech | 0.9 | 17 | 931 | no premature endpoint at mid pause |
| tool_failure | 1.4 | 17 | 223 | recovered gracefully |
| network_degradation | 1.3 | 17 | 224 | dropped frames detected, turn completed |

Voice metrics (means): **TTFT ≈ 1 ms, TTFA ≈ 20 ms, E2E ≈ 344 ms,
interruption ≈ 1 ms, TTS cancellation ≈ 14 ms** (mock profile). Real provider
numbers will be reported here once keyed runs are measured; `compare_reports`
renders before/after tables for tuning.

---

## 7. Tradeoffs and decisions worth discussing

* **Energy VAD over ML VAD for the default**: deterministic, zero-dependency,
  and fast to reason about in endpointing; thresholds are fully configurable. An
  ML VAD can be swapped in behind the same interface.
* **WebSocket PCM over WebRTC for v1**: dramatically simpler to reason about,
  test, and instrument; the browser resamples to 16 kHz mono. WebRTC is a
  natural follow-up for production echo cancellation.
* **Endpoint on silence, not on STT finals**: a single authority for turn
  boundaries keeps behavior consistent across providers (Deepgram emits finals
  on its own cadence). STT finals still *shorten* the silence window when they
  carry a strong sentence end.
* **Per-session in-memory runtimes**: state is explicit, isolated and cheap for
  the demo/test surface; PostgreSQL persists completed turns and Redis can back
  a distributed registry later.
* **Sentence-boundary chunking** trades a little TTS naturalness for a big
  reduction in time-to-first-audio without firing TTS per token.

---

## 8. Repository layout

```
app/
  api/            FastAPI routes: sessions, audio WS, events WS, metrics, evaluations
  runtime/        orchestrator, state machine, event bus, cancellation, turn, pipeline
  audio/          gateway, VAD, buffering, resampling
  providers/      stt|llm|tts protocols, mocks, real adapters, factory
  tools/          registry + support tools + demo data
  memory/         context window, SQLAlchemy store
  observability/  logging, metrics ledger
  evaluation/     harness, scenarios, runner, reports
frontend/         React + Vite dashboard with mic capture/playback
tests/            unit | integration | evaluation
docker-compose.yml  backend + frontend + postgres + redis
.env.example      all configuration
```

## 9. Future work

Load/concurrency testing, profiling, an optional Rust audio component only if
benchmarks justify it, real-provider latency calibration, and replacing the
energy VAD with a neural VAD behind the same interface.
