"""FastAPI application main entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.routing import ASGIRoute
from starlette.responses import Response

from fastmcp.server.streamable_http import StreamableHTTPSessionManager

from .routes import auth, index, notes, search
from ..mcp.server import mcp
from ..services.seed import init_and_seed

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Viewer API",
    description="Multi-tenant Obsidian-like documentation system",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://huggingface.co",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event: Initialize database and seed demo data
@app.on_event("startup")
async def startup_event():
    """Initialize database schema and seed demo vault on startup."""
    logger.info("Running startup: initializing database and seeding demo vault...")
    try:
        init_and_seed(user_id="demo-user")
        logger.info("Startup complete: database and demo vault ready")
    except Exception as e:
        logger.exception(f"Startup failed: {e}")
        # Don't crash the app, but log the error
        logger.error("App starting without demo data due to initialization error")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "detail": str(exc)},
    )


@app.exception_handler(409)
async def conflict_handler(request: Request, exc: Exception):
    """Handle 409 Conflict errors."""
    return JSONResponse(
        status_code=409,
        content={"error": "Conflict", "detail": str(exc)},
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# Mount routers (auth must come first for /auth/login and /auth/callback)
app.include_router(auth.router, tags=["auth"])
app.include_router(notes.router, tags=["notes"])
app.include_router(search.router, tags=["search"])
app.include_router(index.router, tags=["index"])

# Hosted MCP HTTP endpoint (mounted Starlette app)

session_manager = StreamableHTTPSessionManager(
    app=mcp._mcp_server,
    event_store=None,
    json_response=False,
    stateless=True,
)


@app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
async def mcp_http_bridge(request: Request) -> Response:
    """Forward HTTP requests to the FastMCP streamable HTTP session manager."""

    send_queue: asyncio.Queue = asyncio.Queue()

    async def send(message):
        await send_queue.put(message)

    await session_manager.handle_request(request.scope, request.receive, send)
    await send_queue.put(None)

    result_body = b""
    headers = {}
    status = 200

    while True:
        message = await send_queue.get()
        if message is None:
            break
        msg_type = message["type"]
        if msg_type == "http.response.start":
            status = message.get("status", 200)
            raw_headers = message.get("headers", [])
            headers = {key.decode(): value.decode() for key, value in raw_headers}
        elif msg_type == "http.response.body":
            result_body += message.get("body", b"")
            if not message.get("more_body"):
                break

    return Response(content=result_body, status_code=status, headers=headers)


logger.info("MCP HTTP endpoint mounted at /mcp via StreamableHTTPSessionManager")


@app.get("/health")
async def health():
    """Health check endpoint for HF Spaces."""
    return {"status": "healthy"}


# Serve frontend static files with SPA support (must be last to not override API routes)
from fastapi.responses import FileResponse

frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if frontend_dist.exists():
    # Mount static assets
    app.mount(
        "/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets"
    )

    # Catch-all route for SPA - serve index.html for all non-API routes
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the SPA for all non-API routes."""
        # Don't intercept API or auth routes
        if full_path.startswith(("api/", "auth/", "health", "mcp")):
            # Let FastAPI's 404 handler take over
            raise HTTPException(status_code=404, detail="Not found")

        # If the path looks like a file (has extension), try to serve it
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html for SPA routing
        return FileResponse(frontend_dist / "index.html")

    logger.info(f"Serving frontend SPA from: {frontend_dist}")
else:
    logger.warning(f"Frontend dist not found at: {frontend_dist}")

    # Fallback health endpoint if no frontend
    @app.get("/")
    async def root():
        """API health check endpoint."""
        return {"status": "ok", "service": "Document Viewer API"}


__all__ = ["app"]
