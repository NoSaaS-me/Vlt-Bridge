"""API routes for Client communication."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services.daemon_service import DaemonService

router = APIRouter()
daemon_service = DaemonService()

@router.websocket("/ws/client/connect")
async def client_websocket(websocket: WebSocket):
    """WebSocket for Desktop/Web clients."""
    await websocket.accept()
    
    # Simulate sending history on connect
    await websocket.send_text('{"type": "history", "data": "Previous session output..."}')
    
    try:
        while True:
            data = await websocket.receive_text()
            daemons = await daemon_service.get_active_daemons()
            if daemons:
                await websocket.send_text(f"Echo from server: {data}")
            else:
                await websocket.send_text('{"error": "No daemons online"}')
    except WebSocketDisconnect:
        pass