"""FastAPI exception handlers aligned with HTTP API contract."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

DEFAULT_ERRORS: Dict[int, Tuple[str, str]] = {
    status.HTTP_400_BAD_REQUEST: ("validation_error", "Invalid request payload"),
    status.HTTP_401_UNAUTHORIZED: ("unauthorized", "Authorization required"),
    status.HTTP_403_FORBIDDEN: ("forbidden", "Forbidden"),
    status.HTTP_404_NOT_FOUND: ("not_found", "Resource not found"),
    status.HTTP_409_CONFLICT: ("version_conflict", "Resource version conflict"),
    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: (
        "payload_too_large",
        "Payload exceeds allowed size",
    ),
    status.HTTP_500_INTERNAL_SERVER_ERROR: ("internal_error", "Internal server error"),
}


def _normalize_error(
    status_code: int, detail: Any
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    default_error, default_message = DEFAULT_ERRORS.get(
        status_code, DEFAULT_ERRORS[status.HTTP_500_INTERNAL_SERVER_ERROR]
    )
    if isinstance(detail, dict):
        error = detail.get("error", default_error)
        message = detail.get("message", default_message)
        detail_payload = detail.get("detail")
        if detail_payload is None:
            remainder = {
                k: v for k, v in detail.items() if k not in {"error", "message", "detail"}
            }
            detail_payload = remainder or None
        return error, message, detail_payload
    if isinstance(detail, str) and detail:
        return default_error, detail, None
    return default_error, default_message, None


def _response(status_code: int, detail: Any) -> JSONResponse:
    error, message, extra = _normalize_error(status_code, detail)
    return JSONResponse(
        status_code=status_code, content={"error": error, "message": message, "detail": extra}
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    detail = {"detail": {"errors": exc.errors()}}
    return _response(status.HTTP_400_BAD_REQUEST, detail)


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    return _response(exc.status_code, exc.detail)


async def internal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return _response(status.HTTP_500_INTERNAL_SERVER_ERROR, exc.args[0] if exc.args else None)


def register_error_handlers(app: FastAPI) -> None:
    """Attach shared exception handlers to the FastAPI application."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, internal_exception_handler)


__all__ = [
    "register_error_handlers",
    "validation_exception_handler",
    "http_exception_handler",
    "internal_exception_handler",
]
