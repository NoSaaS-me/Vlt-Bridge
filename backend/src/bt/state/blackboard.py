"""
TypedBlackboard - Type-safe hierarchical blackboard for behavior trees.

Part of the BT Universal Runtime (spec 019).
Implements the blackboard.yaml contract.
"""

import json
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .base import (
    BlackboardScope,
    ErrorResult,
    Severity,
    make_reserved_key_error,
    make_schema_validation_error,
    make_size_limit_error,
    make_unregistered_key_error,
)

T = TypeVar("T", bound=BaseModel)

# Configuration from blackboard.yaml
MAX_SIZE_BYTES = 104857600  # 100MB
MAX_KEY_LENGTH = 256
RESERVED_PREFIX = "_"
MAX_LUA_TABLE_DEPTH = 20


class TypedBlackboard:
    """
    Type-safe hierarchical blackboard with schema enforcement.

    Primary state management mechanism for behavior trees.
    Supports scope chain lookup (child -> parent -> ... -> root).

    Invariants:
    - All keys in _data have corresponding entry in _schemas
    - All values in _data are valid instances of their schema
    - Parent chain is acyclic (no circular references)
    - scope_name is non-empty string
    - Total size does not exceed max_size_bytes
    """

    def __init__(
        self,
        parent: Optional["TypedBlackboard"] = None,
        scope_name: str = "root",
    ) -> None:
        """
        Initialize a new blackboard.

        Args:
            parent: Parent blackboard for scope chain lookup. None for root scope.
            scope_name: Human-readable name for this scope.

        Raises:
            ValueError: If scope_name is empty.
            TypeError: If parent is not a TypedBlackboard or None.
        """
        if not scope_name or not isinstance(scope_name, str):
            raise ValueError("scope_name cannot be empty")

        if parent is not None and not isinstance(parent, TypedBlackboard):
            raise TypeError("parent must be TypedBlackboard or None")

        # Check for circular references
        if parent is not None:
            self._check_circular_reference(parent, scope_name)

        self._parent = parent
        self._scope_name = scope_name
        self._data: Dict[str, Any] = {}
        self._schemas: Dict[str, Type[BaseModel]] = {}
        self._reads: Set[str] = set()
        self._writes: Set[str] = set()
        self._size_bytes: int = 0
        self._max_size_bytes: int = MAX_SIZE_BYTES

        # Inherit schemas from parent if any
        if parent is not None:
            self._schemas = dict(parent._schemas)

    def _check_circular_reference(
        self, parent: "TypedBlackboard", new_scope_name: str
    ) -> None:
        """Check for circular references in parent chain."""
        visited = {new_scope_name}
        current = parent
        chain = [new_scope_name]

        while current is not None:
            if current._scope_name in visited:
                chain.append(current._scope_name)
                raise ValueError(
                    f"Circular reference detected in scope chain: {' -> '.join(chain)}"
                )
            visited.add(current._scope_name)
            chain.append(current._scope_name)
            current = current._parent

    # =========================================================================
    # Schema Registration
    # =========================================================================

    def register(self, key: str, schema: Type[BaseModel]) -> None:
        """
        Register expected type for a key.

        Args:
            key: The blackboard key name.
            schema: Pydantic BaseModel subclass for validation.

        Raises:
            ValueError: If key is empty, too long, or starts with reserved prefix.
            TypeError: If schema is not a BaseModel subclass.
        """
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")

        if len(key) > MAX_KEY_LENGTH:
            raise ValueError(f"key too long: {len(key)} > {MAX_KEY_LENGTH}")

        if key.startswith(RESERVED_PREFIX):
            raise ValueError(
                f"Cannot register key '{key}': keys starting with '{RESERVED_PREFIX}' are system-reserved"
            )

        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("schema must be a subclass of pydantic.BaseModel")

        self._schemas[key] = schema

    def register_many(self, schemas: Dict[str, Type[BaseModel]]) -> None:
        """
        Register multiple schemas atomically.

        If any registration fails, none are registered.

        Args:
            schemas: Mapping of key names to Pydantic schemas.

        Raises:
            ValueError: If any key is invalid.
            TypeError: If any schema is not a BaseModel subclass.
        """
        # Validate all first (atomic)
        for key, schema in schemas.items():
            if not key or not isinstance(key, str):
                raise ValueError(f"key must be a non-empty string, got: {key!r}")

            if len(key) > MAX_KEY_LENGTH:
                raise ValueError(f"key too long: {len(key)} > {MAX_KEY_LENGTH}")

            if key.startswith(RESERVED_PREFIX):
                raise ValueError(
                    f"Cannot register key '{key}': keys starting with '{RESERVED_PREFIX}' are system-reserved"
                )

            if not isinstance(schema, type) or not issubclass(schema, BaseModel):
                raise TypeError(
                    f"schema for '{key}' must be a subclass of pydantic.BaseModel"
                )

        # All valid, register all
        for key, schema in schemas.items():
            self._schemas[key] = schema

    # =========================================================================
    # Data Access
    # =========================================================================

    def get(
        self,
        key: str,
        schema: Type[T],
        default: Optional[T] = None,
    ) -> Optional[T]:
        """
        Get typed value with scope chain lookup.

        Lookup order: this scope -> parent -> parent.parent -> ... -> root

        Args:
            key: The blackboard key to retrieve.
            schema: Expected Pydantic schema type.
            default: Value to return if key not found.

        Returns:
            The value if found, otherwise default (or None).

        Raises:
            ValueError: If key is empty.
            RuntimeError: If schema doesn't match registered schema (E1002).
        """
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")

        # Track read
        self._reads.add(key)

        # Check schema mismatch if registered
        registered_schema = self._get_registered_schema(key)
        if registered_schema is not None and registered_schema != schema:
            raise RuntimeError(
                f"E1002: Schema mismatch for key '{key}': "
                f"registered as {registered_schema.__name__}, "
                f"requested as {schema.__name__}"
            )

        # Scope chain lookup
        value = self._lookup(key)
        if value is not None:
            return value

        # Not found - check if we should error or return default
        if default is None and registered_schema is None:
            # Key not registered and no default - could be an error scenario
            # But per contract, we return None (error only if explicitly required)
            pass

        return default

    def _lookup(self, key: str) -> Optional[Any]:
        """Look up key in scope chain."""
        if key in self._data:
            return self._data[key]
        if self._parent is not None:
            return self._parent._lookup(key)
        return None

    def _get_registered_schema(self, key: str) -> Optional[Type[BaseModel]]:
        """Get registered schema from this scope or parents."""
        if key in self._schemas:
            return self._schemas[key]
        if self._parent is not None:
            return self._parent._get_registered_schema(key)
        return None

    def set(
        self,
        key: str,
        value: Union[BaseModel, Dict[str, Any]],
    ) -> ErrorResult[None]:
        """
        Set validated value in current scope.

        Args:
            key: The blackboard key.
            value: The value to set. Must match registered schema.

        Returns:
            ErrorResult indicating success or failure with error details.
        """
        if not key or not isinstance(key, str):
            raise ValueError("key must be a non-empty string")

        # Check reserved key
        if key.startswith(RESERVED_PREFIX):
            error = make_reserved_key_error(key)
            return ErrorResult(success=False, error=error)

        # Check if registered
        schema = self._get_registered_schema(key)
        if schema is None:
            error = make_unregistered_key_error(key, list(self._schemas.keys()))
            return ErrorResult(success=False, error=error)

        # Validate value
        try:
            if isinstance(value, dict):
                validated = schema.model_validate(value)
            elif isinstance(value, BaseModel):
                if not isinstance(value, schema):
                    # Convert to dict and re-validate
                    validated = schema.model_validate(value.model_dump())
                else:
                    validated = value
            else:
                # Try to construct from value
                validated = schema.model_validate(value)
        except ValidationError as e:
            value_preview = str(value)[:100]
            error = make_schema_validation_error(
                key=key,
                expected_schema=schema.__name__,
                actual_type=type(value).__name__,
                validation_error=str(e),
                value_preview=value_preview,
            )
            return ErrorResult(success=False, error=error)

        # Calculate size
        new_size = self._calculate_value_size(validated)
        old_size = self._get_key_size(key)
        delta = new_size - old_size

        # Check size limit
        if self._size_bytes + delta > self._max_size_bytes:
            error = make_size_limit_error(
                current_size_bytes=self._size_bytes + delta,
                limit_bytes=self._max_size_bytes,
                key=key,
                value_size_bytes=new_size,
            )
            return ErrorResult(success=False, error=error)

        # Write to data
        self._data[key] = validated
        self._writes.add(key)
        self._size_bytes += delta

        return ErrorResult.ok()

    def set_internal(self, key: str, value: Any) -> None:
        """
        Set system-reserved key (bypasses underscore check).

        For internal runtime keys like _failure_trace, _parallel_conflicts.

        Args:
            key: Must start with underscore.
            value: Any value (no schema validation).

        Raises:
            ValueError: If key doesn't start with underscore.
        """
        if not key.startswith(RESERVED_PREFIX):
            raise ValueError(f"Internal keys must start with '{RESERVED_PREFIX}'")

        old_size = self._get_key_size(key)
        self._data[key] = value
        self._writes.add(key)

        # Update size tracking
        new_size = self._calculate_value_size(value)
        self._size_bytes += new_size - old_size

    def set_global(
        self,
        key: str,
        value: Union[BaseModel, Dict[str, Any]],
    ) -> ErrorResult[None]:
        """
        Set value in root (global) scope.

        Follows parent chain to top and sets there.

        Args:
            key: The blackboard key.
            value: The value to set.

        Returns:
            ErrorResult from the root scope's set operation.
        """
        # Track write in this scope for tracking purposes
        self._writes.add(key)

        # Find root scope
        root = self._get_root()
        return root.set(key, value)

    def _get_root(self) -> "TypedBlackboard":
        """Get the root (topmost) scope."""
        current = self
        while current._parent is not None:
            current = current._parent
        return current

    def has(self, key: str) -> bool:
        """
        Check if key exists in this scope or parents.

        Does NOT track as a read (peek only).

        Args:
            key: The key to check.

        Returns:
            True if key has a value anywhere in scope chain.
        """
        if key in self._data:
            return True
        if self._parent is not None:
            return self._parent.has(key)
        return False

    def delete(self, key: str) -> bool:
        """
        Delete key from this scope only.

        Parent scope values remain visible after delete in child.

        Args:
            key: The key to delete.

        Returns:
            True if key was present in THIS scope.
        """
        if key in self._data:
            old_size = self._get_key_size(key)
            del self._data[key]
            self._size_bytes -= old_size
            return True
        return False

    # =========================================================================
    # Scope Management
    # =========================================================================

    def create_child_scope(self, scope_name: str) -> "TypedBlackboard":
        """
        Create isolated child scope for parallel children.

        The child:
        - Has self as parent
        - Inherits all schemas from parent chain
        - Has empty _data (no value copying)
        - Has empty _reads and _writes

        Args:
            scope_name: Name for the child scope.

        Returns:
            New TypedBlackboard with self as parent.

        Raises:
            ValueError: If scope_name is empty.
        """
        if not scope_name or not isinstance(scope_name, str):
            raise ValueError("scope_name cannot be empty")

        return TypedBlackboard(parent=self, scope_name=scope_name)

    # =========================================================================
    # Access Tracking
    # =========================================================================

    def get_reads(self) -> Set[str]:
        """Get keys read since last clear."""
        return set(self._reads)

    def get_writes(self) -> Set[str]:
        """Get keys written since last clear."""
        return set(self._writes)

    def clear_access_tracking(self) -> None:
        """Reset read/write tracking. Called at tick start."""
        self._reads = set()
        self._writes = set()

    # =========================================================================
    # Debugging
    # =========================================================================

    def snapshot(self) -> Dict[str, Any]:
        """
        Create serializable snapshot of all data including parents.

        Returns merged dict of all scopes (child wins on conflict).
        Values are Pydantic model_dump() output.

        Returns:
            Dictionary with all blackboard data.
        """
        result = {}

        # Start with parent data (if any)
        if self._parent is not None:
            result = self._parent.snapshot()

        # Overlay our data (child wins on conflict)
        for key, value in self._data.items():
            if isinstance(value, BaseModel):
                result[key] = value.model_dump()
            else:
                result[key] = value

        return result

    def get_size_bytes(self) -> int:
        """Get current size of this scope's data."""
        return self._size_bytes

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information about this blackboard."""
        return {
            "scope_name": self._scope_name,
            "parent_scope": self._parent._scope_name if self._parent else None,
            "size_bytes": self._size_bytes,
            "key_count": len(self._data),
            "registered_schemas": list(self._schemas.keys()),
            "reads_this_tick": list(self._reads),
            "writes_this_tick": list(self._writes),
        }

    # =========================================================================
    # Size Calculation Helpers
    # =========================================================================

    def _calculate_value_size(self, value: Any) -> int:
        """Calculate approximate size of a value in bytes."""
        try:
            if isinstance(value, BaseModel):
                return len(value.model_dump_json().encode("utf-8"))
            else:
                return len(json.dumps(value, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            # Fallback for non-serializable values
            return len(str(value).encode("utf-8"))

    def _get_key_size(self, key: str) -> int:
        """Get the size of an existing key's value, or 0 if not present."""
        if key not in self._data:
            return 0
        return self._calculate_value_size(self._data[key])


# =============================================================================
# Lua Type Coercion Helpers
# =============================================================================


def _is_array_like(table: Any) -> bool:
    """
    Check if a Lua table is array-like.

    A table is array-like if it has sequential integer keys starting at 1
    with no gaps.
    """
    try:
        keys = list(table.keys())
        if not keys:
            return False  # Empty table defaults to dict

        # Check if all keys are integers
        if not all(isinstance(k, (int, float)) for k in keys):
            return False

        # Convert to ints and check for sequential from 1
        int_keys = sorted(int(k) for k in keys)
        expected = list(range(1, len(int_keys) + 1))
        return int_keys == expected
    except (AttributeError, TypeError):
        return False


def lua_to_python(value: Any, depth: int = 0) -> Any:
    """
    Convert Lua value to Python value.

    Type coercion rules per blackboard.yaml:
    - nil -> None
    - boolean -> bool
    - number -> float (ALWAYS, even integers!)
    - string -> str
    - table (array-like) -> list
    - table (dict-like) -> dict
    - function -> NOT ALLOWED (raises ValueError)
    - userdata -> NOT ALLOWED (raises ValueError)

    Args:
        value: Lua value to convert.
        depth: Current recursion depth (for protection).

    Returns:
        Python equivalent value.

    Raises:
        ValueError: If value type is not supported or nesting too deep.
    """
    if depth > MAX_LUA_TABLE_DEPTH:
        raise ValueError(f"Lua table nesting too deep (>{MAX_LUA_TABLE_DEPTH})")

    # None/nil
    if value is None:
        return None

    # Boolean (check before number since bool is subclass of int in Python)
    if isinstance(value, bool):
        return value

    # Number -> float (always!)
    if isinstance(value, (int, float)):
        return float(value)

    # String
    if isinstance(value, str):
        return value

    # Lua table (from lupa)
    # Try to detect lupa table types
    type_name = type(value).__name__
    if "LuaTable" in type_name or "lua" in type_name.lower():
        return _convert_lua_table(value, depth)

    # Check if it's callable (Lua function)
    if callable(value):
        raise ValueError("Lua functions cannot be stored in blackboard (E1002)")

    # Fallback: try dict-like conversion
    if hasattr(value, "items"):
        return {
            str(k): lua_to_python(v, depth + 1)
            for k, v in value.items()
        }

    # Fallback: try list-like conversion
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return [lua_to_python(v, depth + 1) for v in value]

    # Unknown type - let it pass through (might be a Python object)
    return value


def _convert_lua_table(table: Any, depth: int) -> Union[List[Any], Dict[str, Any]]:
    """Convert a Lua table to Python list or dict."""
    if _is_array_like(table):
        # Array-like: convert to list
        return [lua_to_python(v, depth + 1) for v in table.values()]
    else:
        # Dict-like: convert to dict with string keys
        return {
            str(k): lua_to_python(v, depth + 1)
            for k, v in table.items()
        }


def python_to_lua(value: Any, depth: int = 0) -> Any:
    """
    Convert Python value for Lua consumption.

    Type coercion rules per blackboard.yaml:
    - None -> nil (None)
    - bool -> boolean
    - int -> number
    - float -> number
    - str -> string
    - list -> table (array)
    - dict -> table
    - BaseModel -> table (via model_dump())

    Args:
        value: Python value to convert.
        depth: Current recursion depth.

    Returns:
        Value suitable for Lua (Python-native, lupa handles conversion).
    """
    if depth > MAX_LUA_TABLE_DEPTH:
        raise ValueError(f"Nesting too deep (>{MAX_LUA_TABLE_DEPTH})")

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return float(value)  # Lua numbers are always float

    if isinstance(value, str):
        return value

    if isinstance(value, BaseModel):
        # Convert to dict first, then recursively convert
        return python_to_lua(value.model_dump(), depth)

    if isinstance(value, dict):
        return {
            str(k): python_to_lua(v, depth + 1)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [python_to_lua(v, depth + 1) for v in value]

    # Fallback: return as-is
    return value


# Re-export BlackboardScope for convenience
__all__ = [
    "TypedBlackboard",
    "BlackboardScope",
    "lua_to_python",
    "python_to_lua",
    "MAX_SIZE_BYTES",
    "MAX_KEY_LENGTH",
    "RESERVED_PREFIX",
]
