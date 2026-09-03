from __future__ import annotations

import json

import numpy as np
import websockets.sync.client

from app.audio.resampling import float32_to_pcm16_bytes
from app.main import create_app
from tests.conftest import make_settings
from tests.integration.test_api_ws import RunningServer, _create_session, _read_until
from tests.unit.test_vad import silence_samples, tone_samples


def _frame(samples: np.ndarray) -> bytes:
    return float32_to_pcm16_bytes(samples)


def test_agent_audio_streams_back_over_audio_ws() -> None:
    app = create_app(make_settings())
    with RunningServer(app) as (http_base, ws_base):
        session_id = _create_session(http_base)
        with websockets.sync.client.connect(f"{ws_base}/sessions/{session_id}/audio", max_size=2**20) as audio_ws:
            speech = _frame(tone_samples(0.5, amplitude=0.3))
            silence = _frame(silence_samples(0.05))
            frame_bytes = len(speech)
            audio_ws.send(speech)
            for _ in range(9):
                audio_ws.send(speech)
            for _ in range(14):
                audio_ws.send(silence)

            saw_start = False
            saw_audio = 0
            end_reason = None
            for _ in range(4000):
                message = audio_ws.recv(timeout=8)
                if isinstance(message, bytes):
                    saw_audio += len(message) // 2
                    continue
                payload = json.loads(message)
                if payload.get("type") == "agent_audio.start":
                    saw_start = True
                elif payload.get("type") == "agent_audio.end":
                    end_reason = payload.get("reason")
                    break

            assert saw_start is True
            assert saw_audio > 0
            assert frame_bytes > 0
            assert end_reason == "completed"


def test_events_ws_reports_speaking_state_during_agent_audio() -> None:
    app = create_app(make_settings())
    with RunningServer(app) as (http_base, ws_base):
        session_id = _create_session(http_base)
        with websockets.sync.client.connect(f"{ws_base}/sessions/{session_id}/events") as events_ws:
            with websockets.sync.client.connect(f"{ws_base}/sessions/{session_id}/audio", max_size=2**20) as audio_ws:
                for _ in range(25):
                    audio_ws.send(_frame(tone_samples(0.02, amplitude=0.3)))
                for _ in range(30):
                    audio_ws.send(_frame(silence_samples(0.02)))

                messages = _read_until(
                    events_ws,
                    {"transcript_final", "turn_started", "llm_started", "tts_started", "tts_audio"},
                )
                states = [m["to_state"] for m in messages if m.get("type") == "runtime_state_changed"]
                assert "speaking" in states or "processing" in states
