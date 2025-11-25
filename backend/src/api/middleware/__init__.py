"""FastAPI middleware for authentication and error handling."""

from .auth_middleware import AuthContext, extract_user_id_from_jwt, get_auth_context
from .error_handlers import (
    http_exception_handler,
    internal_exception_handler,
    register_error_handlers,
    validation_exception_handler,
)

__all__ = [
    "AuthContext",
    "extract_user_id_from_jwt",
    "get_auth_context",
    "register_error_handlers",
    "validation_exception_handler",
    "http_exception_handler",
    "internal_exception_handler",
]
