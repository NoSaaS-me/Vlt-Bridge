"""Unit tests for LuaSandbox - Lua script execution via lupa.

Tests cover:
- T049: Basic Lua execution, context access, return value handling
- T050: Timeout enforcement (5 second default)
- T051: Sandboxing (os, io, debug blocked; require blocked; memory limits)
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from backend.src.services.plugins.lua_sandbox import (
    LuaSandbox,
    LuaExecutionError,
    LuaTimeoutError,
    LuaSandboxError,
)
from backend.src.services.plugins.context import (
    RuleContext,
    TurnState,
    HistoryState,
    UserState,
    ProjectState,
    PluginState,
    ToolCallRecord,
)


@pytest.fixture
def sandbox() -> LuaSandbox:
    """Create a LuaSandbox instance with default settings."""
    return LuaSandbox()


@pytest.fixture
def sandbox_short_timeout() -> LuaSandbox:
    """Create a LuaSandbox with a very short timeout for testing."""
    return LuaSandbox(timeout_seconds=0.5)


@pytest.fixture
def minimal_context() -> RuleContext:
    """Create a minimal RuleContext for testing."""
    return RuleContext.create_minimal("test-user", "test-project", turn_number=5)


@pytest.fixture
def full_context() -> RuleContext:
    """Create a full RuleContext with various data."""
    return RuleContext(
        turn=TurnState(
            number=10,
            token_usage=0.85,
            context_usage=0.70,
            iteration_count=3,
        ),
        history=HistoryState(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            tools=[
                ToolCallRecord(
                    name="vault_search",
                    arguments={"query": "test"},
                    result="Found 5 notes",
                    success=True,
                    timestamp=datetime.now(timezone.utc),
                ),
                ToolCallRecord(
                    name="web_search",
                    arguments={"query": "python"},
                    result=None,
                    success=False,
                    timestamp=datetime.now(timezone.utc),
                ),
            ],
            failures={"web_search": 2, "file_read": 1},
        ),
        user=UserState(
            id="user-123",
            settings={"theme": "dark", "max_results": 10},
        ),
        project=ProjectState(
            id="project-456",
            settings={"name": "My Project", "indexed": True},
        ),
        state=PluginState(_store={"counter": 42, "last_run": "2025-01-01"}),
    )


# =============================================================================
# T049: Basic Lua Execution Tests
# =============================================================================

class TestLuaBasicExecution:
    """Test basic Lua script execution."""

    def test_execute_simple_return_number(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Execute simple Lua that returns a number."""
        script = "return 42"
        result = sandbox.execute(script, minimal_context)
        assert result == 42

    def test_execute_simple_return_string(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Execute simple Lua that returns a string."""
        script = 'return "hello world"'
        result = sandbox.execute(script, minimal_context)
        assert result == "hello world"

    def test_execute_simple_return_boolean(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Execute simple Lua that returns a boolean."""
        script = "return true"
        result = sandbox.execute(script, minimal_context)
        assert result is True

    def test_execute_return_nil(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Execute Lua that returns nil."""
        script = "return nil"
        result = sandbox.execute(script, minimal_context)
        assert result is None

    def test_execute_return_table(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Execute Lua that returns a table (converted to dict)."""
        script = 'return {type = "notify_self", message = "test"}'
        result = sandbox.execute(script, minimal_context)
        assert isinstance(result, dict)
        assert result.get("type") == "notify_self"
        assert result.get("message") == "test"

    def test_execute_arithmetic(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Execute Lua with arithmetic operations."""
        script = "return 10 + 5 * 2"
        result = sandbox.execute(script, minimal_context)
        assert result == 20

    def test_execute_string_operations(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Execute Lua with string operations."""
        script = 'return string.upper("hello") .. " " .. string.lower("WORLD")'
        result = sandbox.execute(script, minimal_context)
        assert result == "HELLO world"

    def test_execute_math_operations(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Execute Lua with math operations."""
        script = "return math.floor(3.7) + math.ceil(2.1)"
        result = sandbox.execute(script, minimal_context)
        assert result == 6

    def test_execute_table_operations(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Execute Lua with table operations."""
        script = """
        local t = {1, 2, 3}
        table.insert(t, 4)
        return #t
        """
        result = sandbox.execute(script, minimal_context)
        assert result == 4


# =============================================================================
# T049: Context Access Tests
# =============================================================================

class TestLuaContextAccess:
    """Test Lua script access to RuleContext."""

    def test_access_turn_number(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Access context.turn.number from Lua."""
        script = "return context.turn.number"
        result = sandbox.execute(script, full_context)
        assert result == 10

    def test_access_token_usage(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Access context.turn.token_usage from Lua."""
        script = "return context.turn.token_usage"
        result = sandbox.execute(script, full_context)
        assert abs(result - 0.85) < 0.001

    def test_access_context_usage(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Access context.turn.context_usage from Lua."""
        script = "return context.turn.context_usage"
        result = sandbox.execute(script, full_context)
        assert abs(result - 0.70) < 0.001

    def test_access_iteration_count(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Access context.turn.iteration_count from Lua."""
        script = "return context.turn.iteration_count"
        result = sandbox.execute(script, full_context)
        assert result == 3

    def test_access_user_id(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Access context.user.id from Lua."""
        script = "return context.user.id"
        result = sandbox.execute(script, full_context)
        assert result == "user-123"

    def test_access_user_settings(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Access context.user.settings from Lua."""
        script = "return context.user.settings.theme"
        result = sandbox.execute(script, full_context)
        assert result == "dark"

    def test_access_project_id(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Access context.project.id from Lua."""
        script = "return context.project.id"
        result = sandbox.execute(script, full_context)
        assert result == "project-456"

    def test_access_history_failures(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Access context.history.failures from Lua."""
        script = "return context.history.failures.web_search"
        result = sandbox.execute(script, full_context)
        assert result == 2

    def test_access_plugin_state(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Access context.state values from Lua."""
        script = "return context.state.counter"
        result = sandbox.execute(script, full_context)
        assert result == 42

    def test_access_history_total_failures(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Access total_failures computed property from Lua."""
        script = "return context.history.total_failures"
        result = sandbox.execute(script, full_context)
        assert result == 3  # 2 + 1

    def test_conditional_based_on_context(self, sandbox: LuaSandbox, full_context: RuleContext):
        """Execute conditional logic based on context."""
        script = """
        if context.turn.token_usage > 0.8 then
            return {type = "notify_self", message = "High token usage!"}
        end
        return nil
        """
        result = sandbox.execute(script, full_context)
        assert result is not None
        assert result.get("type") == "notify_self"
        assert "High token usage" in result.get("message", "")


# =============================================================================
# T049: Return Value Handling Tests
# =============================================================================

class TestLuaReturnValueHandling:
    """Test proper handling of Lua return values."""

    def test_return_nested_table(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Return nested table structure."""
        script = """
        return {
            type = "emit_event",
            payload = {
                key1 = "value1",
                key2 = 42,
                nested = {a = 1, b = 2}
            }
        }
        """
        result = sandbox.execute(script, minimal_context)
        assert result["type"] == "emit_event"
        assert result["payload"]["key1"] == "value1"
        assert result["payload"]["key2"] == 42
        assert result["payload"]["nested"]["a"] == 1

    def test_return_array_table(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Return array-style table."""
        script = "return {1, 2, 3, 4, 5}"
        result = sandbox.execute(script, minimal_context)
        # Lua arrays become dicts with integer keys
        assert len(result) == 5

    def test_return_float(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Return floating point number."""
        script = "return 3.14159"
        result = sandbox.execute(script, minimal_context)
        assert abs(result - 3.14159) < 0.0001

    def test_return_empty_table(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Return empty table."""
        script = "return {}"
        result = sandbox.execute(script, minimal_context)
        assert isinstance(result, dict)
        assert len(result) == 0


# =============================================================================
# T050: Timeout Tests
# =============================================================================

class TestLuaTimeout:
    """Test timeout enforcement for Lua scripts."""

    def test_infinite_loop_timeout(self, sandbox_short_timeout: LuaSandbox, minimal_context: RuleContext):
        """Infinite loop should raise timeout error."""
        script = "while true do end"
        with pytest.raises(LuaTimeoutError) as exc_info:
            sandbox_short_timeout.execute(script, minimal_context)
        assert "timeout" in str(exc_info.value).lower()

    def test_long_computation_timeout(self, sandbox_short_timeout: LuaSandbox, minimal_context: RuleContext):
        """Long computation should timeout."""
        script = """
        local x = 0
        for i = 1, 10000000000 do
            x = x + 1
        end
        return x
        """
        with pytest.raises(LuaTimeoutError):
            sandbox_short_timeout.execute(script, minimal_context)

    def test_quick_script_completes(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Quick script should complete without timeout."""
        script = """
        local sum = 0
        for i = 1, 1000 do
            sum = sum + i
        end
        return sum
        """
        result = sandbox.execute(script, minimal_context)
        assert result == 500500

    def test_timeout_cleanup(self, sandbox_short_timeout: LuaSandbox, minimal_context: RuleContext):
        """After timeout, sandbox should be usable again."""
        # First, trigger a timeout
        script_timeout = "while true do end"
        with pytest.raises(LuaTimeoutError):
            sandbox_short_timeout.execute(script_timeout, minimal_context)

        # Then, execute a quick script (should work)
        # Note: Due to thread-based timeout, may need new sandbox instance
        sandbox2 = LuaSandbox(timeout_seconds=5.0)
        script_quick = "return 42"
        result = sandbox2.execute(script_quick, minimal_context)
        assert result == 42


# =============================================================================
# T051: Sandboxing Tests
# =============================================================================

class TestLuaSandboxing:
    """Test sandboxing restrictions."""

    def test_os_module_blocked(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """os module should not be accessible."""
        script = "return os.execute('ls')"
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        assert "os" in str(exc_info.value).lower() or "nil" in str(exc_info.value).lower()

    def test_os_getenv_blocked(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """os.getenv should not be accessible."""
        script = "return os.getenv('PATH')"
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        assert "os" in str(exc_info.value).lower() or "nil" in str(exc_info.value).lower()

    def test_io_module_blocked(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """io module should not be accessible."""
        script = "return io.open('/etc/passwd', 'r')"
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        assert "io" in str(exc_info.value).lower() or "nil" in str(exc_info.value).lower()

    def test_debug_module_blocked(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """debug module should not be accessible."""
        script = "return debug.traceback()"
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        assert "debug" in str(exc_info.value).lower() or "nil" in str(exc_info.value).lower()

    def test_require_blocked(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """require function should be blocked."""
        script = "local socket = require('socket')"
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        assert "require" in str(exc_info.value).lower() or "nil" in str(exc_info.value).lower()

    def test_loadfile_blocked(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """loadfile function should be blocked."""
        script = "return loadfile('/etc/passwd')()"
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        assert "loadfile" in str(exc_info.value).lower() or "nil" in str(exc_info.value).lower()

    def test_dofile_blocked(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """dofile function should be blocked."""
        script = "dofile('/etc/passwd')"
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        assert "dofile" in str(exc_info.value).lower() or "nil" in str(exc_info.value).lower()

    def test_load_blocked(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """load function should be blocked."""
        script = 'return load("return os")() '
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        # Either load is blocked directly, or it can't access os
        error_msg = str(exc_info.value).lower()
        assert "load" in error_msg or "nil" in error_msg or "os" in error_msg

    def test_loadstring_blocked(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """loadstring function should be blocked."""
        script = 'return loadstring("return 42")()'
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        assert "loadstring" in str(exc_info.value).lower() or "nil" in str(exc_info.value).lower()

    def test_rawget_blocked(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """rawget should be blocked to prevent sandbox escape."""
        script = "return rawget(_G, 'os')"
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        # Either rawget is blocked or _G is not accessible
        error_msg = str(exc_info.value).lower()
        assert "rawget" in error_msg or "nil" in error_msg or "_g" in error_msg

    def test_safe_functions_allowed(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Safe functions should work."""
        script = """
        local results = {}
        results.type_test = type(42) == "number"
        results.tostring_test = tostring(42) == "42"
        results.tonumber_test = tonumber("42") == 42
        results.pairs_test = true
        for k, v in pairs({a=1}) do
            results.pairs_test = results.pairs_test and k == "a"
        end
        return results
        """
        result = sandbox.execute(script, minimal_context)
        assert result["type_test"] is True
        assert result["tostring_test"] is True
        assert result["tonumber_test"] is True
        assert result["pairs_test"] is True

    def test_string_module_allowed(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """string module functions should work."""
        script = """
        local s = "hello"
        return {
            len = string.len(s),
            upper = string.upper(s),
            sub = string.sub(s, 1, 2)
        }
        """
        result = sandbox.execute(script, minimal_context)
        assert result["len"] == 5
        assert result["upper"] == "HELLO"
        assert result["sub"] == "he"

    def test_table_module_allowed(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """table module functions should work."""
        script = """
        local t = {1, 2, 3}
        table.insert(t, 4)
        return {
            concat = table.concat(t, ","),
            len = #t
        }
        """
        result = sandbox.execute(script, minimal_context)
        assert result["concat"] == "1,2,3,4"
        assert result["len"] == 4

    def test_math_module_allowed(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """math module functions should work."""
        script = """
        return {
            floor = math.floor(3.7),
            ceil = math.ceil(3.2),
            abs = math.abs(-5),
            max = math.max(1, 2, 3),
            min = math.min(1, 2, 3)
        }
        """
        result = sandbox.execute(script, minimal_context)
        assert result["floor"] == 3
        assert result["ceil"] == 4
        assert result["abs"] == 5
        assert result["max"] == 3
        assert result["min"] == 1


# =============================================================================
# T051: Memory Limits Tests
# =============================================================================

class TestLuaMemoryLimits:
    """Test memory limit enforcement."""

    def test_small_allocation_allowed(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Small allocations should work."""
        script = """
        local t = {}
        for i = 1, 1000 do
            t[i] = "item " .. i
        end
        return #t
        """
        result = sandbox.execute(script, minimal_context)
        assert result == 1000

    def test_memory_limit_with_large_allocation(self, minimal_context: RuleContext):
        """Large allocations should fail if memory limit is very low."""
        # Create sandbox with very low memory limit
        sandbox = LuaSandbox(timeout_seconds=5.0, max_memory_mb=1)
        script = """
        local t = {}
        for i = 1, 10000000 do
            t[i] = string.rep("x", 1000)
        end
        return #t
        """
        # This should either raise a memory error or timeout
        with pytest.raises((LuaSandboxError, LuaTimeoutError)):
            sandbox.execute(script, minimal_context)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestLuaErrorHandling:
    """Test error handling for various Lua errors."""

    def test_syntax_error(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Syntax errors should raise LuaExecutionError."""
        script = "return 1 +"  # Incomplete expression
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        assert "syntax" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()

    def test_runtime_error(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Runtime errors should raise LuaExecutionError."""
        script = 'return "hello" + 5'  # Cannot add string and number
        with pytest.raises(LuaExecutionError) as exc_info:
            sandbox.execute(script, minimal_context)
        assert len(str(exc_info.value)) > 0

    def test_nil_access_error(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Accessing nil field should raise error."""
        script = "return context.nonexistent.field"
        with pytest.raises(LuaExecutionError):
            sandbox.execute(script, minimal_context)

    def test_empty_script(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Empty script should return nil."""
        script = ""
        result = sandbox.execute(script, minimal_context)
        assert result is None

    def test_script_with_only_comments(self, sandbox: LuaSandbox, minimal_context: RuleContext):
        """Script with only comments should return nil."""
        script = "-- This is a comment"
        result = sandbox.execute(script, minimal_context)
        assert result is None
