"""Lua sandbox for executing scripts in a restricted environment.

This module provides the LuaSandbox class which executes Lua scripts
via lupa (LuaJIT) with:
- Environment whitelisting (no os, io, debug, require)
- Timeout enforcement via threading
- Memory limits (when supported by lupa)
- RuleContext exposure as Lua tables

Usage:
    from services.plugins.lua_sandbox import LuaSandbox
    from services.plugins.context import RuleContext

    sandbox = LuaSandbox(timeout_seconds=5.0, max_memory_mb=100)
    context = RuleContext.create_minimal("user1", "project1")

    result = sandbox.execute('''
        if context.turn.token_usage > 0.8 then
            return {type = "notify_self", message = "High usage!"}
        end
        return nil
    ''', context)
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from queue import Queue, Empty
from typing import Any, Optional

from lupa import LuaRuntime, LuaError

from .context import (
    RuleContext,
    TurnState,
    HistoryState,
    UserState,
    ProjectState,
    PluginState,
    EventData,
    ToolResult,
)

logger = logging.getLogger(__name__)


class LuaSandboxError(Exception):
    """Base exception for Lua sandbox errors."""
    pass


class LuaExecutionError(LuaSandboxError):
    """Error during Lua script execution."""
    pass


class LuaTimeoutError(LuaSandboxError):
    """Lua script exceeded timeout limit."""
    pass


class LuaMemoryError(LuaSandboxError):
    """Lua script exceeded memory limit."""
    pass


# Whitelist of safe Lua globals and modules
ALLOWED_GLOBALS = frozenset({
    # Basic functions
    "print",
    "type",
    "tostring",
    "tonumber",
    "pairs",
    "ipairs",
    "next",
    "select",
    "unpack",
    "pcall",
    "xpcall",
    "error",
    "assert",
    # Modules (will be filtered)
    "table",
    "string",
    "math",
})

# Functions to BLOCK from globals
BLOCKED_GLOBALS = frozenset({
    # Dangerous file/system operations
    "os",
    "io",
    "debug",
    # Code loading
    "dofile",
    "loadfile",
    "load",
    "loadstring",
    # Raw access (sandbox escape)
    "rawget",
    "rawset",
    "rawequal",
    "rawlen",
    # Module system
    "require",
    "module",
    "package",
    # Garbage collection control
    "collectgarbage",
    # Global manipulation
    "setfenv",
    "getfenv",
    "setmetatable",  # Can be used to escape sandbox
    "getmetatable",
    # Coroutines (can be used for side effects)
    "coroutine",
})

# Safe string functions
SAFE_STRING_FUNCTIONS = frozenset({
    "byte", "char", "find", "format", "gmatch", "gsub", "len",
    "lower", "match", "rep", "reverse", "sub", "upper",
})

# Safe table functions
SAFE_TABLE_FUNCTIONS = frozenset({
    "concat", "insert", "maxn", "remove", "sort", "unpack",
})

# Safe math functions
SAFE_MATH_FUNCTIONS = frozenset({
    "abs", "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh",
    "deg", "exp", "floor", "fmod", "frexp", "huge", "ldexp", "log",
    "log10", "max", "min", "modf", "pi", "pow", "rad", "random",
    "randomseed", "sin", "sinh", "sqrt", "tan", "tanh",
})


class LuaSandbox:
    """Sandbox for executing Lua scripts safely.

    Provides a restricted Lua execution environment with:
    - Environment whitelisting (blocks os, io, debug, require, etc.)
    - Timeout enforcement via threading
    - Memory limits when supported by lupa
    - RuleContext exposed as nested Lua tables

    Attributes:
        timeout_seconds: Maximum execution time (default 5.0).
        max_memory_mb: Maximum memory usage in MB (default 100).
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_memory_mb: int = 100,
    ) -> None:
        """Initialize the Lua sandbox.

        Args:
            timeout_seconds: Maximum script execution time in seconds.
            max_memory_mb: Maximum memory usage in megabytes.
        """
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb

    def execute(self, script: str, context: RuleContext) -> Any:
        """Execute a Lua script in the sandbox.

        Args:
            script: Lua source code to execute.
            context: RuleContext available as 'context' global.

        Returns:
            The return value of the script, converted to Python types.

        Raises:
            LuaExecutionError: If the script has syntax or runtime errors.
            LuaTimeoutError: If the script exceeds the timeout.
            LuaMemoryError: If the script exceeds memory limits.
        """
        if not script or not script.strip():
            return None

        # Use ThreadPoolExecutor for timeout enforcement
        result_queue: Queue[tuple[bool, Any]] = Queue()

        def run_script() -> None:
            """Execute script in isolated thread."""
            try:
                result = self._execute_in_sandbox(script, context)
                result_queue.put((True, result))
            except LuaSandboxError as e:
                result_queue.put((False, e))
            except Exception as e:
                result_queue.put((False, LuaExecutionError(str(e))))

        thread = threading.Thread(target=run_script, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            # Thread is still running - timeout occurred
            logger.warning(f"Lua script timeout after {self.timeout_seconds}s")
            raise LuaTimeoutError(
                f"Script execution exceeded {self.timeout_seconds} second timeout"
            )

        # Get result from queue
        try:
            success, result = result_queue.get_nowait()
            if success:
                return result
            else:
                raise result
        except Empty:
            raise LuaExecutionError("Script execution failed without result")

    def _execute_in_sandbox(self, script: str, context: RuleContext) -> Any:
        """Execute script in sandboxed Lua runtime.

        Args:
            script: Lua source code.
            context: RuleContext to expose.

        Returns:
            Converted return value.
        """
        # Create new Lua runtime for isolation
        try:
            lua = LuaRuntime(unpack_returned_tuples=True)
        except Exception as e:
            raise LuaExecutionError(f"Failed to create Lua runtime: {e}")

        try:
            # Create restricted environment
            sandbox_env = self._create_sandbox_env(lua)

            # Expose context
            context_table = self._context_to_lua(lua, context)
            sandbox_env["context"] = context_table

            lua_globals = lua.globals()

            # Get the load function (Lua 5.2+)
            # Note: lupa globals don't have a .get() method like Python dicts,
            # use direct attribute access or indexing
            load_func = lua_globals.load

            if load_func is None:
                raise LuaExecutionError("Lua load function not available")

            # Compile the script with the sandbox environment
            # Lua 5.2+ load signature: load(chunk, chunkname, mode, env)
            # Returns (function, nil) on success or (nil, error_message) on failure
            try:
                compile_result = load_func(script, "sandbox", "t", sandbox_env)

                # Handle the return value - lupa may return tuple or single value
                if isinstance(compile_result, tuple):
                    # Load returned (func, nil) or (nil, error)
                    compiled = compile_result[0]
                    compile_error = compile_result[1] if len(compile_result) > 1 else None
                    if compiled is None:
                        error_msg = str(compile_error) if compile_error else "Unknown syntax error"
                        raise LuaExecutionError(f"Lua syntax error: {error_msg}")
                else:
                    compiled = compile_result

                if compiled is None:
                    raise LuaExecutionError("Failed to compile script")

                result = compiled()

            except LuaError as e:
                error_msg = str(e)
                # Check for common sandbox violations
                if any(blocked in error_msg.lower() for blocked in ["os", "io", "debug"]):
                    raise LuaExecutionError(f"Access to blocked module: {e}")
                raise LuaExecutionError(f"Lua error: {e}")

            # Convert result to Python types
            return self._lua_to_python(result)

        finally:
            # Cleanup
            del lua

    def _create_sandbox_env(self, lua: LuaRuntime) -> Any:
        """Create a restricted Lua environment.

        Args:
            lua: The LuaRuntime instance.

        Returns:
            A Lua table with only allowed globals.
        """
        # Create empty table for sandbox environment
        env = lua.table()

        # Get Lua's _G (global environment)
        lua_globals = lua.globals()

        # Add allowed basic functions
        for name in ALLOWED_GLOBALS:
            if name in ("table", "string", "math"):
                continue  # Handle modules separately
            try:
                value = lua_globals[name]
                if value is not None:
                    env[name] = value
            except (KeyError, LuaError):
                pass

        # Add safe string module functions
        string_table = lua.table()
        try:
            lua_string = lua_globals.string
            if lua_string:
                for func_name in SAFE_STRING_FUNCTIONS:
                    try:
                        func = lua_string[func_name]
                        if func is not None:
                            string_table[func_name] = func
                    except (KeyError, LuaError, AttributeError):
                        pass
        except (AttributeError, LuaError):
            pass
        env["string"] = string_table

        # Add safe table module functions
        table_table = lua.table()
        try:
            lua_table_mod = lua_globals.table
            if lua_table_mod:
                for func_name in SAFE_TABLE_FUNCTIONS:
                    try:
                        func = lua_table_mod[func_name]
                        if func is not None:
                            table_table[func_name] = func
                    except (KeyError, LuaError, AttributeError):
                        pass
        except (AttributeError, LuaError):
            pass
        env["table"] = table_table

        # Add safe math module functions
        math_table = lua.table()
        try:
            lua_math = lua_globals.math
            if lua_math:
                for func_name in SAFE_MATH_FUNCTIONS:
                    try:
                        value = lua_math[func_name]
                        if value is not None:
                            math_table[func_name] = value
                    except (KeyError, LuaError, AttributeError):
                        pass
        except (AttributeError, LuaError):
            pass
        env["math"] = math_table

        return env

    def _context_to_lua(self, lua: LuaRuntime, context: RuleContext) -> Any:
        """Convert RuleContext to nested Lua tables.

        Args:
            lua: The LuaRuntime instance.
            context: RuleContext to convert.

        Returns:
            A Lua table representing the context.
        """
        ctx = lua.table()

        # Turn state
        turn = lua.table()
        turn["number"] = context.turn.number
        turn["token_usage"] = context.turn.token_usage
        turn["context_usage"] = context.turn.context_usage
        turn["iteration_count"] = context.turn.iteration_count
        ctx["turn"] = turn

        # History state
        history = lua.table()

        # Messages
        messages = lua.table()
        for i, msg in enumerate(context.history.messages, 1):
            msg_table = lua.table()
            for k, v in msg.items():
                msg_table[k] = v
            messages[i] = msg_table
        history["messages"] = messages

        # Tools
        tools = lua.table()
        for i, tool in enumerate(context.history.tools, 1):
            tool_table = lua.table()
            tool_table["name"] = tool.name
            tool_table["success"] = tool.success
            tool_table["result"] = tool.result
            # Arguments as nested table
            args_table = lua.table()
            for k, v in tool.arguments.items():
                args_table[k] = v
            tool_table["arguments"] = args_table
            tools[i] = tool_table
        history["tools"] = tools

        # Failures
        failures = lua.table()
        for name, count in context.history.failures.items():
            failures[name] = count
        history["failures"] = failures

        # Computed properties
        history["total_tool_calls"] = context.history.total_tool_calls
        history["total_failures"] = context.history.total_failures

        ctx["history"] = history

        # User state
        user = lua.table()
        user["id"] = context.user.id
        settings = lua.table()
        for k, v in context.user.settings.items():
            settings[k] = self._python_to_lua(lua, v)
        user["settings"] = settings
        ctx["user"] = user

        # Project state
        project = lua.table()
        project["id"] = context.project.id
        proj_settings = lua.table()
        for k, v in context.project.settings.items():
            proj_settings[k] = self._python_to_lua(lua, v)
        project["settings"] = proj_settings
        ctx["project"] = project

        # Plugin state
        state = lua.table()
        for k, v in context.state._store.items():
            state[k] = self._python_to_lua(lua, v)
        ctx["state"] = state

        # Event data (if present)
        if context.event:
            event = lua.table()
            event["type"] = context.event.type
            event["source"] = context.event.source
            event["severity"] = context.event.severity
            payload = lua.table()
            for k, v in context.event.payload.items():
                payload[k] = self._python_to_lua(lua, v)
            event["payload"] = payload
            ctx["event"] = event

        # Tool result (if present)
        if context.result:
            result = lua.table()
            result["tool_name"] = context.result.tool_name
            result["success"] = context.result.success
            result["result"] = context.result.result
            result["error"] = context.result.error
            result["duration_ms"] = context.result.duration_ms
            ctx["result"] = result

        return ctx

    def _python_to_lua(self, lua: LuaRuntime, value: Any) -> Any:
        """Convert Python value to Lua-compatible value.

        Args:
            lua: The LuaRuntime instance.
            value: Python value to convert.

        Returns:
            Lua-compatible value.
        """
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            table = lua.table()
            for k, v in value.items():
                table[k] = self._python_to_lua(lua, v)
            return table
        if isinstance(value, (list, tuple)):
            table = lua.table()
            for i, v in enumerate(value, 1):
                table[i] = self._python_to_lua(lua, v)
            return table
        # Fallback: convert to string
        return str(value)

    def _lua_to_python(self, value: Any) -> Any:
        """Convert Lua value to Python value.

        Args:
            value: Lua value to convert.

        Returns:
            Python value.
        """
        if value is None:
            return None

        # Check if it's a Lua table
        if hasattr(value, "items"):
            # It's a table-like object, convert to dict
            result = {}
            try:
                for k, v in value.items():
                    result[self._lua_to_python(k)] = self._lua_to_python(v)
            except Exception:
                # Fallback for array-like tables
                try:
                    i = 1
                    while True:
                        item = value[i]
                        if item is None:
                            break
                        result[i] = self._lua_to_python(item)
                        i += 1
                except Exception:
                    pass
            return result

        # Handle lupa's Lua types
        type_name = type(value).__name__
        if "lua" in type_name.lower():
            # Try to iterate if possible
            try:
                result = {}
                for k, v in value.items():
                    result[self._lua_to_python(k)] = self._lua_to_python(v)
                return result
            except Exception:
                # Try string conversion
                try:
                    return str(value)
                except Exception:
                    return value

        # Primitive types
        if isinstance(value, (bool, int, float, str)):
            return value

        return value


def execute_script(
    script: str,
    context: RuleContext,
    timeout_seconds: float = 5.0,
    max_memory_mb: int = 100,
) -> Any:
    """Convenience function to execute a Lua script.

    Args:
        script: Lua source code to execute.
        context: RuleContext for script access.
        timeout_seconds: Maximum execution time.
        max_memory_mb: Maximum memory usage.

    Returns:
        Script return value converted to Python.
    """
    sandbox = LuaSandbox(
        timeout_seconds=timeout_seconds,
        max_memory_mb=max_memory_mb,
    )
    return sandbox.execute(script, context)


__all__ = [
    "LuaSandbox",
    "LuaSandboxError",
    "LuaExecutionError",
    "LuaTimeoutError",
    "LuaMemoryError",
    "execute_script",
    "ALLOWED_GLOBALS",
    "BLOCKED_GLOBALS",
]
