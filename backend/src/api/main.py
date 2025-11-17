"""FastAPI application main entry point."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import index, notes, search

app = FastAPI(
    title="Document Viewer API",
    description="Multi-tenant Obsidian-like documentation system",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# Mount routers
app.include_router(notes.router, tags=["notes"])
app.include_router(search.router, tags=["search"])
app.include_router(index.router, tags=["index"])


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Document Viewer API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


__all__ = ["app"]

