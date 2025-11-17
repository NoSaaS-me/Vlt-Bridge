"""FastAPI application main entry point."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

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

# Hosted MCP HTTP endpoint (mounted Starlette app)
mcp_http_app = mcp.http_app(path="/", transport="http")
app.mount("/mcp", mcp_http_app)
logger.info("MCP HTTP endpoint mounted at /mcp")


@app.get("/health")
async def health():
    """Health check endpoint for HF Spaces."""
    return {"status": "healthy"}


# Serve frontend static files with SPA support (must be last to not override API routes)
from fastapi.responses import FileResponse

frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if frontend_dist.exists():
    # Mount static assets
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")
    
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

