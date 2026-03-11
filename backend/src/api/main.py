"""FastAPI application main entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()  # Add this line at the top, before other imports

# from fastapi.routing import ASGIRoute
from starlette.responses import Response

from fastmcp.server.http import StreamableHTTPSessionManager, set_http_request
from fastapi.responses import FileResponse

from .routes import auth, index, notes, search, graph, system, rag, tts, models, oracle, oracle_context, threads, projects, coderag, notifications, rules, settings, assets, connectors, connector_oauth, connector_webhooks, composio_hub, admin
from .middleware import SecurityHeadersMiddleware
from ..mcp.server import mcp
from ..services.database import DatabaseService
from ..services.config import get_config

# Oracle V2 imports (023-oracle-codeact-rework) — conditional on package availability
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as _AsyncSqliteSaver
    _ORACLE_V2_AVAILABLE = True
except ImportError:
    _AsyncSqliteSaver = None
    _ORACLE_V2_AVAILABLE = False

try:
    from graphiti_core import Graphiti as _Graphiti
    _GRAPHITI_AVAILABLE = True
except ImportError:
    _Graphiti = None
    _GRAPHITI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Hosted MCP HTTP endpoint (mounted Starlette app)
session_manager = StreamableHTTPSessionManager(
    app=mcp._mcp_server,
    event_store=None,
    json_response=False,
    stateless=True,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler to run startup tasks."""
    logger.info("Running startup: initializing database...")
    try:
        DatabaseService().initialize()
        logger.info("Startup complete: database ready")
    except Exception as exc:
        logger.exception("Startup failed: %s", exc)
        logger.error("App starting with database initialization error")

    # Security warning for ENABLE_NOAUTH_MCP
    if config.enable_noauth_mcp:
        logger.warning("=" * 80)
        logger.warning("⚠️  SECURITY WARNING: ENABLE_NOAUTH_MCP IS ENABLED ⚠️")
        logger.warning("=" * 80)
        logger.warning("The server is running in INSECURE MODE!")
        logger.warning("ENABLE_NOAUTH_MCP bypasses authentication on all routes.")
        logger.warning("This should ONLY be used in isolated development environments.")
        logger.warning("NEVER enable this in production or publicly accessible deployments.")
        logger.warning("=" * 80)

    # Oracle V2: Initialize AsyncSqliteSaver (LangGraph thread checkpointer)
    if _ORACLE_V2_AVAILABLE:
        try:
            from ..services.oracle_v2.graph import _PickleSerde
            _checkpointer_cm = _AsyncSqliteSaver.from_conn_string(config.oracle_checkpoint_db)
            app.state.oracle_checkpointer = await _checkpointer_cm.__aenter__()
            # Override serde: CodeAct stores function objects in REPL context
            # which msgpack (default) can't serialize — pickle handles them.
            app.state.oracle_checkpointer.serde = _PickleSerde()
            logger.info("OracleV2: AsyncSqliteSaver initialized (%s)", config.oracle_checkpoint_db)
        except Exception as exc:
            logger.warning("OracleV2: AsyncSqliteSaver init failed — oracle_v2 disabled: %s", exc)
            app.state.oracle_checkpointer = None
            _checkpointer_cm = None
    else:
        app.state.oracle_checkpointer = None
        _checkpointer_cm = None
        logger.info("OracleV2: langgraph-checkpoint-sqlite not installed, oracle_v2 disabled")

    # Oracle V2: Initialize Graphiti client (cross-session memory)
    if _GRAPHITI_AVAILABLE:
        try:
            app.state.graphiti = _Graphiti(config.falkordb_url, "", "")
            logger.info("OracleV2: Graphiti client connected (%s)", config.falkordb_url)
        except Exception as exc:
            logger.warning("OracleV2: Graphiti init failed — memory disabled: %s", exc)
            app.state.graphiti = None
    else:
        app.state.graphiti = None
        logger.info("OracleV2: graphiti-core not installed, memory disabled")

    # Initialize FastMCP session manager task group
    async with session_manager.run():
        yield

    # Oracle V2 cleanup
    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning("OracleV2: AsyncSqliteSaver cleanup error: %s", exc)


app = FastAPI(
    title="Document Viewer API",
    description="Multi-tenant Obsidian-like documentation system",
    version="0.1.0",
    lifespan=lifespan,
)

config = get_config()

# CORS middleware — build allowed origins list, filtering out empty/falsy values
_cors_origins = [
    origin for origin in [
        "http://localhost:5173",
        "http://localhost:3000",
        config.chatgpt_cors_origin,
    ]
    if origin and origin.strip() and origin.lower() not in ("none", "null", "")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers middleware (added after CORS so headers appear in response)
app.add_middleware(
    SecurityHeadersMiddleware,
    csp_policy=config.csp_policy,
    frame_options=config.frame_options,
    enable_hsts=config.enable_hsts,
)


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
app.include_router(graph.router, tags=["graph"])
app.include_router(system.router, tags=["system"])
app.include_router(rag.router, tags=["rag"])
app.include_router(tts.router, tags=["tts"])
app.include_router(models.router, tags=["models"])
app.include_router(oracle.router, tags=["oracle"])
app.include_router(oracle_context.router, tags=["oracle-context"])
app.include_router(threads.router, tags=["threads"])
app.include_router(projects.router, tags=["projects"])
app.include_router(coderag.router, tags=["coderag"])
app.include_router(notifications.router, tags=["notifications"])
app.include_router(rules.router, tags=["rules"])
app.include_router(settings.router, tags=["settings"])
app.include_router(assets.router, tags=["assets"])
app.include_router(connectors.router, tags=["connectors"])
app.include_router(connector_oauth.router, tags=["connector-oauth"])
app.include_router(connector_webhooks.router, tags=["webhooks"])
app.include_router(composio_hub.router, tags=["composio-hub"])
app.include_router(admin.router, tags=["admin"])


@app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
async def mcp_http_bridge(request: Request) -> Response:
    """Forward HTTP requests to the FastMCP streamable HTTP session manager."""

    send_queue: asyncio.Queue = asyncio.Queue()

    async def send(message):
        await send_queue.put(message)

    try:
        with set_http_request(request):
            await session_manager.handle_request(request.scope, request.receive, send)
    except Exception as exc:
        logger.exception("FastMCP session manager crashed: %s", exc)
        raise HTTPException(status_code=500, detail=f"MCP Bridge Error: {exc}")

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
        if (
            full_path.startswith(("api/", "auth/"))
            or full_path == "health"
            or full_path.startswith("mcp/")
            or full_path == "mcp"
        ):
            # Let FastAPI's 404 handler take over
            raise HTTPException(status_code=404, detail="Not found")

        # Serve widget entry point
        if full_path == "widget.html" or full_path.startswith("widget"):
            widget_path = frontend_dist / "widget.html"
            if widget_path.is_file():
                # ChatGPT requires specific MIME type for widgets
                return FileResponse(widget_path, media_type="text/html+skybridge")
            logger.warning("widget.html requested but not found")

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