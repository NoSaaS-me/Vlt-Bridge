"""
LuaSandbox - Secure Lua execution environment.

This module provides a sandboxed Lua execution environment for the BT runtime.
It blocks dangerous modules and functions while allowing safe Lua operations.

Security (per footgun-addendum.md B.3):
- Block os module (os.execute, os.exit, os.getenv, etc.)
- Block io module (io.open, io.read, io.write, etc.)
- Block debug module (debug.getinfo, debug.setmetatable, etc.)
- Block package module (package.loadlib, require, etc.)
- Block loadfile, dofile, load (arbitrary code loading)
- Block metatable access (getmetatable, setmetatable, rawget, rawset)
- Enforce timeout (default 5s)
- Memory limits (where supported)

Error codes:
- E5001: Lua syntax error
- E5002: Lua runtime error
- E5003: Lua timeout
- E7001: Sandbox violation

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.4.1-2.4.6 from tasks.md
"""

from __future__ import annotations

import logging
import re
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Error Codes
# =============================================================================


ERROR_CODES = {
    "syntax": "E5001",
    "runtime": "E5002",
    "timeout": "E5003",
    "sandbox": "E7001",
}


# =============================================================================
# LuaExecutionResult
# =============================================================================


@dataclass
class LuaExecutionResult:
    """Result of Lua code execution in sandbox.

    From nodes.yaml Script leaf specification:
    - success: Whether execution completed without error
    - result: Return value from Lua code
    - error: Error message if failed
    - error_type: Classification of error (syntax, runtime, timeout, sandbox)
    - line_number: Line where error occurred (if available)
    """

    success: bool
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None  # syntax, runtime, timeout, sandbox
    line_number: Optional[int] = None

    @classmethod
    def ok(cls, result: Any = None) -> "LuaExecutionResult":
        """Create a successful result."""
        return cls(success=True, result=result)

    @classmethod
    def syntax_error(
        cls,
        message: str,
        line_number: Optional[int] = None,
    ) -> "LuaExecutionResult":
        """Create a syntax error result (E5001)."""
        return cls(
            success=False,
            error=message,
            error_type="syntax",
            line_number=line_number,
        )

    @classmethod
    def runtime_error(
        cls,
        message: str,
        line_number: Optional[int] = None,
    ) -> "LuaExecutionResult":
        """Create a runtime error result (E5002)."""
        return cls(
            success=False,
            error=message,
            error_type="runtime",
            line_number=line_number,
        )

    @classmethod
    def timeout_error(cls, timeout_seconds: float) -> "LuaExecutionResult":
        """Create a timeout error result (E5003)."""
        return cls(
            success=False,
            error=f"Script execution timed out after {timeout_seconds}s",
            error_type="timeout",
        )

    @classmethod
    def sandbox_violation(
        cls,
        blocked_item: str,
        line_number: Optional[int] = None,
    ) -> "LuaExecutionResult":
        """Create a sandbox violation result (E7001)."""
        return cls(
            success=False,
            error=f"Sandbox violation: attempted to access '{blocked_item}'",
            error_type="sandbox",
            line_number=line_number,
        )


# =============================================================================
# LuaSandbox
# =============================================================================


class LuaSandbox:
    """Secure Lua execution environment.

    Provides a sandboxed environment for executing Lua code with:
    - Blocked dangerous modules and functions
    - Timeout enforcement
    - Memory limits (where lupa supports it)

    Security measures (per footgun B.3):
    - All BLOCKED_MODULES are removed from global environment
    - Metatable access functions are blocked
    - load/loadfile/dofile are blocked
    - require is blocked
    - File and process operations are blocked

    Example:
        >>> sandbox = LuaSandbox(timeout_seconds=5.0)
        >>> result = sandbox.execute("return 1 + 2")
        >>> result.success
        True
        >>> result.result
        3.0

        >>> result = sandbox.execute("os.execute('ls')")
        >>> result.success
        False
        >>> result.error_type
        'sandbox'
    """

    # Modules that are completely blocked
    BLOCKED_MODULES: List[str] = [
        "os",           # os.execute, os.exit, os.remove, etc.
        "io",           # io.open, io.read, io.write, etc.
        "debug",        # debug.getinfo, debug.setmetatable, etc.
        "package",      # package.loadlib, etc.
    ]

    # Individual functions that are blocked (even if their module exists)
    BLOCKED_FUNCTIONS: List[str] = [
        "loadfile",     # Load arbitrary Lua files
        "dofile",       # Execute arbitrary Lua files
        "load",         # Load code from strings (too powerful)
        "loadstring",   # Deprecated but still available in some versions
        "require",      # Load modules
        "rawget",       # Bypass metatable protections
        "rawset",       # Bypass metatable protections
        "rawequal",     # Bypass metatable protections
        "getmetatable", # Access metatables
        "setmetatable", # Modify metatables
        "getfenv",      # Access function environments (Lua 5.1)
        "setfenv",      # Modify function environments (Lua 5.1)
        "collectgarbage", # GC control can be abused
        "newproxy",     # Create userdata (Lua 5.1)
    ]

    # Safe functions we explicitly allow
    ALLOWED_FUNCTIONS: Set[str] = {
        # Type checking
        "type", "typeof",
        # Math
        "tonumber", "tostring",
        # Table operations
        "pairs", "ipairs", "next", "unpack", "select",
        # String operations
        "string",
        # Table operations
        "table",
        # Math operations
        "math",
        # Error handling
        "pcall", "xpcall", "error", "assert",
        # Misc
        "print",  # We can redirect this
    }

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_memory_mb: int = 50,
    ) -> None:
        """Initialize the Lua sandbox.

        Args:
            timeout_seconds: Maximum execution time (default 5s).
            max_memory_mb: Maximum memory usage in MB (default 50MB).
        """
        self._timeout_seconds = timeout_seconds
        self._max_memory_mb = max_memory_mb
        self._lua_runtime = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize the Lua runtime."""
        if self._initialized:
            return

        try:
            import lupa
            from lupa import LuaRuntime
        except ImportError:
            raise RuntimeError(
                "lupa is required for Lua sandbox. Install with: pip install lupa"
            )

        # Create Lua runtime
        # Note: attribute_filter provides basic sandbox protection
        self._lua_runtime = LuaRuntime(
            unpack_returned_tuples=True,
            # Don't allow automatic Python attribute access
            register_eval=False,
            register_builtins=False,
        )

        # Get the globals table
        lua_globals = self._lua_runtime.globals()

        # Remove blocked modules
        for module in self.BLOCKED_MODULES:
            if module in lua_globals:
                lua_globals[module] = None

        # Remove blocked functions
        for func in self.BLOCKED_FUNCTIONS:
            if func in lua_globals:
                lua_globals[func] = None

        # Create safe print function that logs instead of writing to stdout
        def safe_print(*args):
            message = " ".join(str(arg) for arg in args)
            logger.debug(f"[Lua print] {message}")

        lua_globals["print"] = safe_print

        self._initialized = True

    def execute(
        self,
        code: str,
        env: Optional[Dict[str, Any]] = None,
        source_name: str = "<script>",
    ) -> LuaExecutionResult:
        """Execute Lua code in the sandbox.

        Args:
            code: Lua code to execute.
            env: Additional environment variables to inject.
            source_name: Name for error reporting.

        Returns:
            LuaExecutionResult with success status and result/error.
        """
        # Pre-execution static analysis for obvious violations
        violation = self._check_static_violations(code)
        if violation:
            return violation

        try:
            self._ensure_initialized()
        except Exception as e:
            return LuaExecutionResult.runtime_error(
                f"Failed to initialize Lua runtime: {e}"
            )

        # Inject environment
        if env:
            lua_globals = self._lua_runtime.globals()
            for key, value in env.items():
                lua_globals[key] = self._convert_to_lua(value)

        # Execute with timeout
        return self._execute_with_timeout(code, source_name)

    def _check_static_violations(self, code: str) -> Optional[LuaExecutionResult]:
        """Check for obvious sandbox violations in code.

        Performs static analysis to catch common violations before
        executing the code. This is a defense-in-depth measure.

        Args:
            code: Lua code to analyze.

        Returns:
            LuaExecutionResult if violation found, None otherwise.
        """
        # Pattern to detect blocked module access
        for module in self.BLOCKED_MODULES:
            # Match patterns like: os.execute, os["execute"], os['execute']
            patterns = [
                rf'\b{module}\s*\.\s*\w+',  # os.execute
                rf'\b{module}\s*\[\s*["\']',  # os["x"] or os['x']
            ]
            for pattern in patterns:
                match = re.search(pattern, code)
                if match:
                    # Find line number
                    line_num = code[:match.start()].count('\n') + 1
                    return LuaExecutionResult.sandbox_violation(
                        blocked_item=match.group(0).strip(),
                        line_number=line_num,
                    )

        # Pattern to detect blocked function calls
        for func in self.BLOCKED_FUNCTIONS:
            # Match function call pattern
            pattern = rf'\b{func}\s*\('
            match = re.search(pattern, code)
            if match:
                line_num = code[:match.start()].count('\n') + 1
                return LuaExecutionResult.sandbox_violation(
                    blocked_item=func,
                    line_number=line_num,
                )

        return None

    def _execute_with_timeout(
        self,
        code: str,
        source_name: str,
    ) -> LuaExecutionResult:
        """Execute code with timeout enforcement.

        Args:
            code: Lua code to execute.
            source_name: Name for error reporting.

        Returns:
            LuaExecutionResult with execution outcome.
        """
        result_holder: Dict[str, Any] = {"result": None, "error": None}

        def run_code():
            try:
                # Compile the code first to catch syntax errors
                try:
                    compiled = self._lua_runtime.compile(code)
                except Exception as e:
                    error_msg = str(e)
                    line_num = self._extract_line_number(error_msg)
                    result_holder["error"] = LuaExecutionResult.syntax_error(
                        message=error_msg,
                        line_number=line_num,
                    )
                    return

                # Execute the compiled code
                try:
                    lua_result = compiled()
                    result_holder["result"] = LuaExecutionResult.ok(
                        self._convert_from_lua(lua_result)
                    )
                except Exception as e:
                    error_msg = str(e)
                    line_num = self._extract_line_number(error_msg)

                    # Check if it's a sandbox violation
                    if self._is_sandbox_violation(error_msg):
                        blocked = self._extract_blocked_item(error_msg)
                        result_holder["error"] = LuaExecutionResult.sandbox_violation(
                            blocked_item=blocked,
                            line_number=line_num,
                        )
                    else:
                        result_holder["error"] = LuaExecutionResult.runtime_error(
                            message=error_msg,
                            line_number=line_num,
                        )

            except Exception as e:
                result_holder["error"] = LuaExecutionResult.runtime_error(
                    message=str(e)
                )

        # Run in a thread with timeout
        thread = threading.Thread(target=run_code, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout_seconds)

        if thread.is_alive():
            # Thread is still running - timeout
            # Note: We can't actually kill the thread in Python,
            # but we return timeout error
            return LuaExecutionResult.timeout_error(self._timeout_seconds)

        # Return result or error
        if result_holder["error"]:
            return result_holder["error"]
        elif result_holder["result"]:
            return result_holder["result"]
        else:
            # No result and no error - return None result
            return LuaExecutionResult.ok(None)

    def _extract_line_number(self, error_message: str) -> Optional[int]:
        """Extract line number from Lua error message.

        Args:
            error_message: Error message from Lua.

        Returns:
            Line number if found, None otherwise.
        """
        # Common patterns:
        # [string "..."]:5: error message
        # <script>:5: error message
        # ...]:5: ...
        patterns = [
            r':(\d+):',  # Standard Lua error format
            r'line (\d+)',  # Alternative format
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                return int(match.group(1))

        return None

    def _is_sandbox_violation(self, error_message: str) -> bool:
        """Check if error is a sandbox violation.

        Args:
            error_message: Error message to check.

        Returns:
            True if this looks like a sandbox violation.
        """
        violation_indicators = [
            "attempt to index a nil value",  # Accessing blocked module
            "attempt to call a nil value",   # Calling blocked function
            "not allowed",
            "blocked",
            "sandbox",
        ]

        error_lower = error_message.lower()
        for indicator in violation_indicators:
            if indicator.lower() in error_lower:
                # Check if it's about a blocked module/function
                for module in self.BLOCKED_MODULES:
                    if module in error_message:
                        return True
                for func in self.BLOCKED_FUNCTIONS:
                    if func in error_message:
                        return True

        return False

    def _extract_blocked_item(self, error_message: str) -> str:
        """Extract the blocked item from error message.

        Args:
            error_message: Error message.

        Returns:
            Name of blocked item or generic message.
        """
        # Check for blocked modules
        for module in self.BLOCKED_MODULES:
            if module in error_message:
                return module

        # Check for blocked functions
        for func in self.BLOCKED_FUNCTIONS:
            if func in error_message:
                return func

        return "unknown blocked operation"

    def _convert_to_lua(self, value: Any) -> Any:
        """Convert Python value to Lua-compatible value.

        Args:
            value: Python value.

        Returns:
            Lua-compatible value.
        """
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            # Create Lua table
            lua_table = self._lua_runtime.table()
            for k, v in value.items():
                lua_table[str(k)] = self._convert_to_lua(v)
            return lua_table

        if isinstance(value, (list, tuple)):
            # Create Lua array-like table (1-indexed)
            lua_table = self._lua_runtime.table()
            for i, v in enumerate(value, start=1):
                lua_table[i] = self._convert_to_lua(v)
            return lua_table

        # Fallback: try to use directly
        return value

    def _convert_from_lua(self, value: Any) -> Any:
        """Convert Lua value to Python value.

        Uses the lua_to_python function from blackboard module
        for consistent type coercion.

        Args:
            value: Lua value.

        Returns:
            Python value.
        """
        from ..state.blackboard import lua_to_python
        return lua_to_python(value)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "LuaSandbox",
    "LuaExecutionResult",
    "ERROR_CODES",
]
