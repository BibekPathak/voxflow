export function backendUrl() {
  const configured = import.meta.env.VITE_BACKEND_URL;
  if (configured) return configured.replace(/\/$/, "");
  const proto = window.location.protocol === "https:" ? "https" : "http";
  return `${proto}://${window.location.hostname}:8000`;
}

export function wsUrl() {
  return backendUrl().replace(/^http/, "ws");
}

const TARGET_RATE = 16000;

export class MicStreamer {
  constructor(onPcm) {
    this.onPcm = onPcm;
    this.audioContext = null;
    this.sourceNode = null;
    this.scriptNode = null;
    this.stream = null;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    this.audioContext = new AudioCtx();
    this.sourceNode = this.audioContext.createMediaStreamSource(this.stream);

    const bufferSize = 4096;
    this.scriptNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
    this.scriptNode.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const pcm = this._process(input, this.audioContext.sampleRate);
      if (pcm.byteLength > 0 && this.onPcm) this.onPcm(pcm);
    };
    this.sourceNode.connect(this.scriptNode);
    this.scriptNode.connect(this.audioContext.destination);
  }

  _process(floatInput, inputRate) {
    const ratio = TARGET_RATE / inputRate;
    const outputLength = Math.floor(floatInput.length * ratio);
    const output = new Int16Array(outputLength);
    for (let i = 0; i < outputLength; i += 1) {
      const sample = floatInput[Math.floor(i / ratio)] || 0;
      const clamped = Math.max(-1, Math.min(1, sample));
      output[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
    return output.buffer;
  }

  stop() {
    if (this.scriptNode) {
      this.scriptNode.onaudioprocess = null;
      this.scriptNode.disconnect();
      this.scriptNode = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

export class AgentAudioPlayer {
  constructor() {
    this.audioContext = null;
    this.queue = [];
    this.nextTime = 0;
    this.currentSource = null;
    this.started = false;
  }

  ensureContext() {
    if (!this.audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioCtx();
    }
    if (this.audioContext.state === "suspended") {
      this.audioContext.resume();
    }
    return this.audioContext;
  }

  reset() {
    this.flush();
  }

  flush() {
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch (err) {
        /* already stopped */
      }
      this.currentSource.disconnect();
      this.currentSource = null;
    }
    this.queue = [];
    this.nextTime = 0;
    this.started = false;
  }

  async push(pcmBytes) {
    const ctx = this.ensureContext();
    const float = new Float32Array(pcmBytes.byteLength / 2);
    const view = new DataView(pcmBytes);
    for (let i = 0; i < float.length; i += 1) {
      float[i] = view.getInt16(i * 2, true) / 32768;
    }
    const buffer = ctx.createBuffer(1, float.length, TARGET_RATE);
    buffer.copyToChannel(float, 0);
    this.queue.push(buffer);
    this._schedule();
  }

  _schedule() {
    const ctx = this.ensureContext();
    if (!this.started) {
      this.nextTime = ctx.currentTime + 0.05;
      this.started = true;
    }
    while (this.queue.length > 0) {
      if (this.nextTime > ctx.currentTime + 0.6) break;
      const buffer = this.queue.shift();
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.start(this.nextTime);
      this.nextTime += buffer.duration;
      this.currentSource = source;
      source.onended = () => {
        if (this.currentSource === source) this.currentSource = null;
      };
    }
  }
}
