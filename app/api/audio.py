from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.runtime.session import SessionManager

router = APIRouter(tags=["audio"])


def _manager(websocket: WebSocket) -> SessionManager:
    return websocket.scope["app"].state.manager


class _WsOutbound:
    """Serializes outbound writes on the audio WebSocket.

    Inbound frames are raw user PCM; outbound frames are agent PCM bytes plus
    JSON control messages (agent_audio.start / agent_audio.end). A lock keeps
    frame ordering intact if audio and control messages are produced from
    concurrent tasks.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()

    async def send_text(self, data: str) -> None:
        async with self._lock:
            await self._websocket.send_text(data)

    async def send_bytes(self, data: bytes) -> None:
        async with self._lock:
            await self._websocket.send_bytes(data)


@router.websocket("/sessions/{session_id}/audio")
async def session_audio_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    try:
        manager = _manager(websocket)
        runtime = manager.get(session_id)
    except Exception:
        await websocket.close(code=4404, reason="session not found")
        return

    owner = uuid.uuid4().hex
    if not runtime.attach(owner, outbound=_WsOutbound(websocket)):
        await websocket.close(code=4409, reason="audio already bound or session closed")
        return

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            payload = message.get("bytes")
            if payload:
                await runtime.ingest_audio(payload)
    except WebSocketDisconnect:
        pass
    finally:
        await runtime.detach(owner)
