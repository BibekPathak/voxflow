from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request

import uvicorn
import websockets.sync.client
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.audio.resampling import float32_to_pcm16_bytes
from app.main import create_app
from tests.conftest import make_settings
from tests.unit.test_vad import silence_samples, tone_samples


def _pcm(samples) -> bytes:
    return float32_to_pcm16_bytes(samples)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class RunningServer:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.port = _free_port()
        self.config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> tuple[str, str]:
        self.thread.start()
        http_url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(http_url, timeout=1) as response:
                    if response.status == 200:
                        http_base = f"http://127.0.0.1:{self.port}"
                        ws_base = f"ws://127.0.0.1:{self.port}"
                        return http_base, ws_base
            except Exception:
                time.sleep(0.05)
        raise TimeoutError("uvicorn server did not start")

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def _create_session(http_base: str) -> str:
    request = urllib.request.Request(
        f"{http_base}/sessions",
        method="POST",
        data=b"",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())["session_id"]


def _read_until(ws, wanted: set[str], max_messages: int = 500) -> list[dict]:
    seen: list[dict] = []
    for _ in range(max_messages):
        message = json.loads(ws.recv(timeout=5))
        seen.append(message)
        if wanted.issubset({m["type"] for m in seen}):
            break
    return seen


def test_session_lifecycle_over_websockets() -> None:
    app = create_app(make_settings())
    with RunningServer(app) as (http_base, ws_base):
        session_id = _create_session(http_base)
        with websockets.sync.client.connect(f"{ws_base}/sessions/{session_id}/events") as events_ws:
            with websockets.sync.client.connect(f"{ws_base}/sessions/{session_id}/audio") as audio_ws:
                audio = _pcm(tone_samples(0.3, amplitude=0.3)) + _pcm(silence_samples(0.3))
                audio_ws.send(audio)
                messages = _read_until(
                    events_ws,
                    {"audio_received", "speech_started", "speech_ended", "runtime_state_changed"},
                )

        types = [m["type"] for m in messages]
        order = [t for t in types if t in {"audio_received", "speech_started", "speech_ended"}]
        assert order.index("speech_started") < order.index("speech_ended")
        states = [m["to_state"] for m in messages if m["type"] == "runtime_state_changed"]
        assert "listening" in states


def test_events_ws_streams_closed_on_disconnect() -> None:
    app = create_app(make_settings())
    with RunningServer(app) as (http_base, ws_base):
        session_id = _create_session(http_base)
        with websockets.sync.client.connect(f"{ws_base}/sessions/{session_id}/events") as events_ws:
            with websockets.sync.client.connect(f"{ws_base}/sessions/{session_id}/audio"):
                pass
            closed_state = None
            for _ in range(100):
                message = json.loads(events_ws.recv(timeout=5))
                if message.get("to_state") == "closed":
                    closed_state = message
                    break
        assert closed_state is not None


def test_get_unknown_session_returns_404() -> None:
    with TestClient(create_app(make_settings())) as client:
        response = client.get("/sessions/does-not-exist")
        assert response.status_code == 404
        assert response.json()["code"] == "SESSION_NOT_FOUND"


def test_audio_bind_conflict_rejected() -> None:
    app = create_app(make_settings())
    with RunningServer(app) as (http_base, ws_base):
        session_id = _create_session(http_base)
        with websockets.sync.client.connect(f"{ws_base}/sessions/{session_id}/audio"):
            try:
                with websockets.sync.client.connect(f"{ws_base}/sessions/{session_id}/audio") as second:
                    second.recv(timeout=3)
                rejected = False
            except websockets.exceptions.ConnectionClosed as exc:
                rejected = exc.rcvd is not None and exc.rcvd.code == 4409
            assert rejected


def test_session_list_returns_created_session() -> None:
    with TestClient(create_app(make_settings())) as client:
        created = client.post("/sessions").json()
        listing = client.get("/sessions").json()
        assert any(s["session_id"] == created["session_id"] for s in listing)
