"""API routes for Daemon communication."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from ..services.daemon_service import DaemonService
from ..models.daemon import DaemonCreate

router = APIRouter()
daemon_service = DaemonService()

@router.websocket("/ws/daemon/{daemon_id}")
async def daemon_websocket(websocket: WebSocket, daemon_id: str):
    """Persistent connection for Vlt Daemons."""
    await websocket.accept()
    try:
        # Register on connect
        await daemon_service.register_daemon(daemon_id, {"status": "connected"})
        print(f"Daemon {daemon_id} connected")
        
        while True:
            data = await websocket.receive_text()
            # Echo for now, actual RPC dispatch later
            await websocket.send_text(f"Ack: {len(data)} bytes")
            
    except WebSocketDisconnect:
        print(f"Daemon {daemon_id} disconnected")
        # TODO: Mark offline in service