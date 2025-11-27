"""System routes for logs and diagnostics."""

import logging
from collections import deque
from typing import List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..middleware import AuthContext, get_auth_context
from ...services.config import PROJECT_ROOT
from fastapi.responses import PlainTextResponse

router = APIRouter()

# Global in-memory log buffer
LOG_BUFFER: deque = deque(maxlen=100)

class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    extra: Dict[str, Any]

class MemoryLogHandler(logging.Handler):
    """Custom handler to capture logs into memory."""
    def emit(self, record):
        try:
            msg = self.format(record)
            extra = {k: v for k, v in record.__dict__.items() 
                     if k not in {'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename', 
                                  'funcName', 'levelname', 'levelno', 'lineno', 'module', 
                                  'msecs', 'message', 'msg', 'name', 'pathname', 'process', 
                                  'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'}}
            
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": msg,
                "extra": extra
            }
            LOG_BUFFER.append(entry)
        except Exception:
            self.handleError(record)

# Attach handler to root logger or specific loggers
memory_handler = MemoryLogHandler()
formatter = logging.Formatter('%(message)s')
memory_handler.setFormatter(formatter)

# Attach to root logger to capture everything
logging.getLogger().addHandler(memory_handler)
# Ensure level allows INFO
logging.getLogger().setLevel(logging.INFO)

@router.get("/api/system/logs", response_model=List[LogEntry])
async def get_logs(auth: AuthContext = Depends(get_auth_context)):
    """Retrieve recent system logs."""
    return list(LOG_BUFFER)

@router.get("/api/system/debug/widget", response_class=PlainTextResponse)
async def debug_widget():
    """Return raw widget.html content for debugging."""
    widget_path = PROJECT_ROOT / "frontend" / "dist" / "widget.html"
    if not widget_path.exists():
        return f"File not found: {widget_path}"
    return widget_path.read_text(encoding="utf-8")
