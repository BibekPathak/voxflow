import { useCallback, useEffect, useRef, useState } from "react";
import { AgentAudioPlayer, MicStreamer, backendUrl, wsUrl } from "./audio.js";

const PIPELINE_KEYS = {
  vad: ["audio_received", "speech_started", "speech_ended"],
  stt: ["transcript_partial", "transcript_final"],
  llm: ["llm_started", "llm_token", "llm_completed"],
  tools: ["tool_call_started", "tool_call_completed", "tool_call_failed"],
  tts: ["tts_started", "tts_audio", "tts_completed"],
};

function pipelineKeysFor(type) {
  return Object.keys(PIPELINE_KEYS).filter((key) => PIPELINE_KEYS[key].includes(type));
}

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [connected, setConnected] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [state, setState] = useState("idle");
  const [messages, setMessages] = useState([]);
  const [partial, setPartial] = useState("");
  const [tools, setTools] = useState([]);
  const [activity, setActivity] = useState({});
  const [latencies, setLatencies] = useState({});
  const [counters, setCounters] = useState({});
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const eventsWsRef = useRef(null);
  const audioWsRef = useRef(null);
  const micRef = useRef(null);
  const playerRef = useRef(new AgentAudioPlayer());

  const noteActivity = useCallback((type) => {
    const keys = pipelineKeysFor(type);
    if (keys.length === 0) return;
    const now = Date.now();
    setActivity((prev) => {
      const next = { ...prev };
      keys.forEach((key) => (next[key] = now));
      return next;
    });
  }, []);

  const handleEvent = useCallback((event) => {
    const type = event.type;
    noteActivity(type);
    switch (type) {
      case "runtime_state_changed":
        setState(event.to_state);
        if (event.to_state === "speaking") setSpeaking(true);
        if (event.to_state === "listening" || event.to_state === "interrupted") setSpeaking(false);
        break;
      case "transcript_partial":
        setPartial(event.text || "");
        break;
      case "transcript_final":
        setPartial("");
        if (event.text) setMessages((m) => [...m, { id: crypto.randomUUID(), role: "user", text: event.text }]);
        break;
      case "turn_started":
        setMessages((m) => [...m, { id: crypto.randomUUID(), role: "agent", text: "" }]);
        break;
      case "llm_token":
        if (event.text) {
          setMessages((m) => {
            const next = [...m];
            const last = next[next.length - 1];
            if (last && last.role === "agent") {
              next[next.length - 1] = { ...last, text: last.text + event.text };
            }
            return next;
          });
        }
        break;
      case "tool_call_started":
        setTools((list) => [
          { id: event.call_id || crypto.randomUUID(), name: event.tool_name, status: "running", arguments: event.arguments },
          ...list,
        ]);
        break;
      case "tool_call_completed":
        setTools((list) =>
          list.map((t) =>
            t.id === event.call_id ? { ...t, status: "done", duration_ms: event.duration_ms } : t
          )
        );
        break;
      case "tool_call_failed":
        setTools((list) =>
          list.map((t) =>
            t.id === event.call_id ? { ...t, status: "failed", error: event.error } : t
          )
        );
        break;
      case "turn_completed":
        if (event.outcome === "cancelled") {
          setSpeaking(false);
        }
        break;
      default:
        break;
    }
  }, [noteActivity]);

  const openSession = useCallback(async () => {
    setBusy(true);
    setError(null);
    setMessages([]);
    setTools([]);
    setLatencies({});
    setCounters({});
    setPartial("");
    try {
      const response = await fetch(`${backendUrl()}/sessions`, { method: "POST" });
      const body = await response.json();
      const id = body.session_id;
      setSessionId(id);
      setConversationId(body.conversation_id);
      setState(body.state);

      const player = playerRef.current;

      const audioWs = new WebSocket(`${wsUrl()}/sessions/${id}/audio`);
      audioWs.binaryType = "arraybuffer";
      audioWs.onopen = () => setConnected(true);
      audioWs.onmessage = async (message) => {
        if (typeof message.data === "string") {
          const control = JSON.parse(message.data);
          if (control.type === "agent_audio.start") {
            player.reset();
            setSpeaking(true);
          } else if (control.type === "agent_audio.end") {
            if (control.reason !== "completed") player.flush();
            setSpeaking(false);
          }
        } else if (message.data instanceof ArrayBuffer) {
          await player.push(message.data);
        }
      };
      audioWs.onclose = () => {
        setConnected(false);
        player.flush();
      };
      audioWsRef.current = audioWs;

      const eventsWs = new WebSocket(`${wsUrl()}/sessions/${id}/events`);
      eventsWs.onmessage = (message) => {
        try {
          handleEvent(JSON.parse(message.data));
        } catch (err) {
          /* ignore malformed */
        }
      };
      eventsWsRef.current = eventsWs;

      const mic = new MicStreamer((pcm) => {
        if (audioWs.readyState === WebSocket.OPEN) audioWs.send(pcm);
      });
      await mic.start();
      micRef.current = mic;
    } catch (err) {
      setError(`Could not start the voice session: ${err.message || err}`);
    } finally {
      setBusy(false);
    }
  }, [handleEvent]);

  const closeSession = useCallback(() => {
    if (micRef.current) {
      micRef.current.stop();
      micRef.current = null;
    }
    [audioWsRef.current, eventsWsRef.current].forEach((socket) => {
      if (socket) socket.close();
    });
    audioWsRef.current = null;
    eventsWsRef.current = null;
    playerRef.current.flush();
    setSpeaking(false);
    setConnected(false);
    setState("idle");
    setSessionId(null);
  }, []);

  useEffect(() => {
    if (!sessionId) return undefined;
    const timer = setInterval(async () => {
      try {
        const response = await fetch(`${backendUrl()}/sessions/${sessionId}/metrics`);
        const body = await response.json();
        const metrics = body.metrics;
        const pick = (name) => {
          const value = metrics.latencies_ms?.[name]?.median;
          return Number.isFinite(value) ? Math.round(value) : null;
        };
        setLatencies({
          ttft: pick("ttft"),
          ttfa: pick("ttfa"),
          e2e: pick("e2e"),
          interruption: pick("interruption"),
          cancellation: pick("tts_cancellation"),
        });
        setCounters(metrics.counters || {});
      } catch (err) {
        /* session may be gone */
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [sessionId]);

  const active = (key) => (activity[key] ? Date.now() - activity[key] < 3000 : false);
  const pipelineStatus = {
    VAD: active("vad"),
    STT: active("stt"),
    LLM: active("llm"),
    TOOLS: active("tools"),
    TTS: active("tts"),
  };

  const metric = (value) => (value == null ? "—" : `${value} ms`);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">VoxFlow <span className="sub">real-time voice agent runtime</span></div>
        <div className="controls">
          {!sessionId ? (
            <button className="primary" disabled={busy} onClick={openSession}>
              Start conversation
            </button>
          ) : (
            <button className="danger" onClick={closeSession}>End session</button>
          )}
        </div>
      </header>

      {error && <div className="banner error">{error}</div>}

      <main className="grid">
        <section className="panel conversation">
          <h2>Live conversation</h2>
          <div className="chat">
            {messages.length === 0 && <div className="empty">Speak to begin. Partial transcripts appear live.</div>}
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.role}`}>
                <span className="who">{message.role === "user" ? "User" : "Agent"}</span>
                <span className="text">{message.text || (message.role === "agent" ? "…" : "")}</span>
              </div>
            ))}
            {partial && <div className="message user partial"><span className="who">User</span><span className="text muted">{partial}</span></div>}
          </div>
        </section>

        <section className="panel runtime">
          <h2>Runtime state</h2>
          <div className="state-row">
            <span className={`state-pill ${state} ${speaking ? "speaking" : ""}`}>{state.toUpperCase()}</span>
            <span className={`mic ${connected ? "live" : ""}`}>{connected ? "AUDIO LINK" : "OFFLINE"}</span>
          </div>
          <h2>Pipeline</h2>
          <div className="pipeline">
            {Object.entries(pipelineStatus).map(([name, isActive]) => (
              <div key={name} className={`stage ${isActive ? "active" : ""}`}>{name}</div>
            ))}
          </div>
          <h2>Latency</h2>
          <div className="latency">
            <div><span>TTFA</span><b>{metric(latencies.ttfa)}</b></div>
            <div><span>TTFT</span><b>{metric(latencies.ttft)}</b></div>
            <div><span>E2E</span><b>{metric(latencies.e2e)}</b></div>
            <div><span>Barge-in</span><b>{metric(latencies.interruption)}</b></div>
            <div><span>TTS cancel</span><b>{metric(latencies.cancellation)}</b></div>
          </div>
          <h2>Session</h2>
          <div className="session-meta">
            <div><span>turns</span><b>{counters.turns_completed || 0}</b></div>
            <div><span>tools</span><b>{counters.tool_calls || 0}</b></div>
            <div><span>interrupts</span><b>{counters.user_interrupts || 0}</b></div>
            {sessionId && <div className="id">session {sessionId.slice(0, 8)}</div>}
          </div>
        </section>

        <section className="panel tools">
          <h2>Tool calls</h2>
          {tools.length === 0 && <div className="empty">No tools invoked yet.</div>}
          {tools.map((tool) => (
            <div key={tool.id} className={`tool ${tool.status}`}>
              <span className="name">{tool.name}</span>
              <span className="status">{tool.status}</span>
              {tool.duration_ms != null && <span className="duration">{Math.round(tool.duration_ms)} ms</span>}
              {tool.arguments && <code>{tool.arguments}</code>}
              {tool.error && <span className="error-text">{tool.error}</span>}
            </div>
          ))}
        </section>
      </main>
      <footer className="foot">VoxFlow mock pipeline — speak into the microphone and interrupt the agent any time.</footer>
    </div>
  );
}
