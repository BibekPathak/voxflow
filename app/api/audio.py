from __future__ import annotations

import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.runtime.session import SessionManager

router = APIRouter(tags=["audio"])


def _manager(websocket: WebSocket) -> SessionManager:
    return websocket.scope["app"].state.manager


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
    if not runtime.attach(owner):
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
