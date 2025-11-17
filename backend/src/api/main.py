"""FastAPI application main entry point."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .routes import auth, index, notes, search
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
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://huggingface.co"],
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

# Mount MCP HTTP endpoint
try:
    from ..mcp.server import mcp
    app.mount("/mcp", mcp.get_asgi_app())
    logger.info("MCP HTTP endpoint mounted at /mcp")
except Exception as e:
    logger.warning(f"Failed to mount MCP HTTP endpoint: {e}")


@app.get("/health")
async def health():
    """Health check endpoint for HF Spaces."""
    return {"status": "healthy"}


# Serve frontend static files (must be last to not override API routes)
frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
    logger.info(f"Serving frontend from: {frontend_dist}")
else:
    logger.warning(f"Frontend dist not found at: {frontend_dist}")
    
    # Fallback health endpoint if no frontend
    @app.get("/")
    async def root():
        """API health check endpoint."""
        return {"status": "ok", "service": "Document Viewer API"}


__all__ = ["app"]

