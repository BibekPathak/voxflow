from __future__ import annotations

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from app.runtime.session import SessionManager

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _manager(request: Request) -> SessionManager:
    return request.app.state.manager


def _manager_from_scope(websocket: WebSocket) -> SessionManager:
    app = websocket.scope["app"]
    return app.state.manager


@router.post("", status_code=201)
async def create_session(request: Request) -> dict:
    manager = _manager(request)
    runtime = manager.create_session()
    return manager.snapshot(runtime)


@router.get("")
async def list_sessions(request: Request) -> list[dict]:
    manager = _manager(request)
    return [manager.snapshot(runtime) for runtime in manager.list()]


@router.get("/{session_id}")
async def get_session(request: Request, session_id: str) -> dict:
    manager = _manager(request)
    runtime = manager.get(session_id)
    return manager.snapshot(runtime)


@router.get("/{session_id}/metrics")
async def get_session_metrics(request: Request, session_id: str) -> dict:
    manager = _manager(request)
    runtime = manager.get(session_id)
    metrics = runtime.metrics.snapshot()
    detector = runtime.detector.stats
    gateway = {
        "speech_active": runtime.gateway.vad.speech_active,
        "total_speech_ms": runtime.gateway.vad.total_speech_ms,
        "total_samples_in": runtime.gateway.total_samples_in,
        "dropped_frames": runtime.gateway.sequencer.total_gaps,
        "missing_frames": runtime.gateway.sequencer.missing_frames,
    }
    return {
        "session_id": session_id,
        "conversation_id": runtime.conversation_id,
        "metrics": metrics,
        "detector": detector,
        "gateway": gateway,
    }


@router.websocket("/{session_id}/events")
async def session_events_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    try:
        manager = _manager_from_scope(websocket)
        runtime = manager.get(session_id)
    except Exception:
        await websocket.close(code=4404, reason="session not found")
        return

    async def emit(event) -> None:
        await websocket.send_text(event.model_dump_json())

    subscription = runtime.bus.subscribe(emit, session_id=session_id)
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        subscription.close()
