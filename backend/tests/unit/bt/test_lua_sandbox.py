"""
Unit tests for LuaSandbox.

Tests the secure Lua execution environment from lua/sandbox.py:
- Successful execution
- Syntax error handling (E5001)
- Runtime error handling (E5002)
- Timeout handling (E5003)
- Sandbox security (E7001) - per footgun B.3

Security tests (footgun B.3):
- test_sandbox_blocks_os_execute
- test_sandbox_blocks_io_open
- test_sandbox_blocks_loadfile
- test_sandbox_blocks_dofile
- test_sandbox_blocks_require
- test_sandbox_blocks_metatable_access

Part of the BT Universal Runtime (spec 019).
"""

import pytest
from unittest.mock import MagicMock, patch
import sys

from backend.src.bt.lua.sandbox import LuaSandbox, LuaExecutionResult, ERROR_CODES


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sandbox() -> LuaSandbox:
    """Create a test sandbox with short timeout."""
    return LuaSandbox(timeout_seconds=2.0, max_memory_mb=10)


# =============================================================================
# Helper to check if lupa is available
# =============================================================================


def lupa_available() -> bool:
    """Check if lupa is installed."""
    try:
        import lupa
        return True
    except ImportError:
        return False


# Mark for skipping if lupa not available
requires_lupa = pytest.mark.skipif(
    not lupa_available(),
    reason="lupa not installed"
)


# =============================================================================
# LuaExecutionResult Tests
# =============================================================================


class TestLuaExecutionResult:
    """Tests for LuaExecutionResult dataclass."""

    def test_ok_result(self) -> None:
        """Test creating successful result."""
        result = LuaExecutionResult.ok(42)

        assert result.success is True
        assert result.result == 42
        assert result.error is None
        assert result.error_type is None

    def test_syntax_error_result(self) -> None:
        """Test creating syntax error result (E5001)."""
        result = LuaExecutionResult.syntax_error(
            message="unexpected symbol",
            line_number=5,
        )

        assert result.success is False
        assert result.error == "unexpected symbol"
        assert result.error_type == "syntax"
        assert result.line_number == 5

    def test_runtime_error_result(self) -> None:
        """Test creating runtime error result (E5002)."""
        result = LuaExecutionResult.runtime_error(
            message="attempt to index nil",
            line_number=10,
        )

        assert result.success is False
        assert result.error == "attempt to index nil"
        assert result.error_type == "runtime"
        assert result.line_number == 10

    def test_timeout_error_result(self) -> None:
        """Test creating timeout error result (E5003)."""
        result = LuaExecutionResult.timeout_error(5.0)

        assert result.success is False
        assert "timed out" in result.error
        assert "5" in result.error
        assert result.error_type == "timeout"

    def test_sandbox_violation_result(self) -> None:
        """Test creating sandbox violation result (E7001)."""
        result = LuaExecutionResult.sandbox_violation(
            blocked_item="os.execute",
            line_number=3,
        )

        assert result.success is False
        assert "Sandbox violation" in result.error
        assert "os.execute" in result.error
        assert result.error_type == "sandbox"
        assert result.line_number == 3


# =============================================================================
# Sandbox Static Analysis Tests
# =============================================================================


class TestSandboxStaticAnalysis:
    """Tests for sandbox static code analysis."""

    def test_detects_os_module_access(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect os.* access."""
        result = sandbox._check_static_violations("os.execute('ls')")

        assert result is not None
        assert result.error_type == "sandbox"
        assert "os" in result.error

    def test_detects_io_module_access(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect io.* access."""
        result = sandbox._check_static_violations("io.open('/etc/passwd')")

        assert result is not None
        assert result.error_type == "sandbox"
        assert "io" in result.error

    def test_detects_debug_module_access(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect debug.* access."""
        result = sandbox._check_static_violations("debug.getinfo(1)")

        assert result is not None
        assert result.error_type == "sandbox"
        assert "debug" in result.error

    def test_detects_loadfile_call(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect loadfile() call."""
        result = sandbox._check_static_violations("loadfile('malicious.lua')")

        assert result is not None
        assert result.error_type == "sandbox"
        assert "loadfile" in result.error

    def test_detects_require_call(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect require() call."""
        result = sandbox._check_static_violations("require('os')")

        assert result is not None
        assert result.error_type == "sandbox"
        assert "require" in result.error

    def test_detects_dofile_call(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect dofile() call."""
        result = sandbox._check_static_violations("dofile('script.lua')")

        assert result is not None
        assert result.error_type == "sandbox"
        assert "dofile" in result.error

    def test_detects_getmetatable_call(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect getmetatable() call."""
        result = sandbox._check_static_violations("getmetatable(t)")

        assert result is not None
        assert result.error_type == "sandbox"
        assert "getmetatable" in result.error

    def test_detects_setmetatable_call(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect setmetatable() call."""
        result = sandbox._check_static_violations("setmetatable(t, {})")

        assert result is not None
        assert result.error_type == "sandbox"
        assert "setmetatable" in result.error

    def test_detects_rawget_call(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect rawget() call."""
        result = sandbox._check_static_violations("rawget(t, 'key')")

        assert result is not None
        assert result.error_type == "sandbox"
        assert "rawget" in result.error

    def test_detects_rawset_call(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect rawset() call."""
        result = sandbox._check_static_violations("rawset(t, 'key', 'value')")

        assert result is not None
        assert result.error_type == "sandbox"
        assert "rawset" in result.error

    def test_allows_safe_code(self, sandbox: LuaSandbox) -> None:
        """Static analysis should allow safe code."""
        safe_codes = [
            "return 1 + 2",
            "local x = 10",
            "for i = 1, 10 do print(i) end",
            "if true then return 1 end",
            "local t = {a = 1, b = 2}",
            "return math.sqrt(16)",
            "return string.upper('hello')",
            "return table.concat({'a', 'b'}, ',')",
        ]

        for code in safe_codes:
            result = sandbox._check_static_violations(code)
            assert result is None, f"Safe code was flagged: {code}"

    def test_detects_bracket_notation_access(self, sandbox: LuaSandbox) -> None:
        """Static analysis should detect os['execute'] style access."""
        result = sandbox._check_static_violations("os['execute']('ls')")

        assert result is not None
        assert result.error_type == "sandbox"

    def test_includes_line_number(self, sandbox: LuaSandbox) -> None:
        """Static analysis should include line number in violation."""
        code = """
local x = 1
local y = 2
os.execute('ls')
return x + y
"""
        result = sandbox._check_static_violations(code)

        assert result is not None
        assert result.line_number == 4  # os.execute is on line 4


# =============================================================================
# Sandbox Execution Tests (require lupa)
# =============================================================================


@requires_lupa
class TestSandboxExecution:
    """Tests for sandbox code execution (requires lupa)."""

    def test_simple_arithmetic(self, sandbox: LuaSandbox) -> None:
        """Sandbox should execute simple arithmetic."""
        result = sandbox.execute("return 1 + 2")

        assert result.success is True
        assert result.result == 3.0  # Lua numbers are floats

    def test_string_operations(self, sandbox: LuaSandbox) -> None:
        """Sandbox should allow string operations."""
        result = sandbox.execute("return string.upper('hello')")

        assert result.success is True
        assert result.result == "HELLO"

    def test_math_operations(self, sandbox: LuaSandbox) -> None:
        """Sandbox should allow math operations."""
        result = sandbox.execute("return math.sqrt(16)")

        assert result.success is True
        assert result.result == 4.0

    def test_table_operations(self, sandbox: LuaSandbox) -> None:
        """Sandbox should allow table operations."""
        result = sandbox.execute("""
            local t = {1, 2, 3}
            return #t
        """)

        assert result.success is True
        assert result.result == 3.0

    def test_nil_return(self, sandbox: LuaSandbox) -> None:
        """Sandbox should handle nil returns."""
        result = sandbox.execute("return nil")

        assert result.success is True
        assert result.result is None

    def test_multiple_returns(self, sandbox: LuaSandbox) -> None:
        """Sandbox should handle multiple returns."""
        result = sandbox.execute("return 1, 2, 3")

        assert result.success is True
        # lupa returns first value when multiple

    def test_table_return(self, sandbox: LuaSandbox) -> None:
        """Sandbox should handle table returns."""
        result = sandbox.execute("return {status = 'success', value = 42}")

        assert result.success is True
        assert isinstance(result.result, dict)
        assert result.result.get("status") == "success"
        assert result.result.get("value") == 42.0

    def test_syntax_error(self, sandbox: LuaSandbox) -> None:
        """Sandbox should catch syntax errors (E5001)."""
        result = sandbox.execute("return {{{{")

        assert result.success is False
        assert result.error_type == "syntax"

    def test_runtime_error(self, sandbox: LuaSandbox) -> None:
        """Sandbox should catch runtime errors (E5002)."""
        result = sandbox.execute("return undefined_variable.field")

        assert result.success is False
        assert result.error_type == "runtime"

    def test_environment_injection(self, sandbox: LuaSandbox) -> None:
        """Sandbox should allow environment injection."""
        result = sandbox.execute(
            "return injected_value * 2",
            env={"injected_value": 21},
        )

        assert result.success is True
        assert result.result == 42.0

    def test_nested_table_injection(self, sandbox: LuaSandbox) -> None:
        """Sandbox should handle nested table injection."""
        result = sandbox.execute(
            "return bb.value + 10",
            env={"bb": {"value": 32}},
        )

        assert result.success is True
        assert result.result == 42.0


# =============================================================================
# Security Tests (footgun B.3)
# =============================================================================


@requires_lupa
class TestSandboxSecurity:
    """Security tests per footgun-addendum.md B.3."""

    def test_sandbox_blocks_os_execute(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block os.execute (B.3)."""
        result = sandbox.execute("os.execute('ls')")

        assert result.success is False
        assert result.error_type == "sandbox"
        assert "os" in result.error.lower()

    def test_sandbox_blocks_os_getenv(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block os.getenv."""
        result = sandbox.execute("return os.getenv('HOME')")

        assert result.success is False
        assert result.error_type == "sandbox"

    def test_sandbox_blocks_io_open(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block io.open (B.3)."""
        result = sandbox.execute("io.open('/etc/passwd', 'r')")

        assert result.success is False
        assert result.error_type == "sandbox"
        assert "io" in result.error.lower()

    def test_sandbox_blocks_io_popen(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block io.popen."""
        result = sandbox.execute("io.popen('ls')")

        assert result.success is False
        assert result.error_type == "sandbox"

    def test_sandbox_blocks_loadfile(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block loadfile (B.3)."""
        result = sandbox.execute("loadfile('/etc/passwd')")

        assert result.success is False
        assert result.error_type == "sandbox"
        assert "loadfile" in result.error.lower()

    def test_sandbox_blocks_dofile(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block dofile (B.3)."""
        result = sandbox.execute("dofile('/etc/passwd')")

        assert result.success is False
        assert result.error_type == "sandbox"
        assert "dofile" in result.error.lower()

    def test_sandbox_blocks_require(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block require (B.3)."""
        result = sandbox.execute("require('os')")

        assert result.success is False
        assert result.error_type == "sandbox"
        assert "require" in result.error.lower()

    def test_sandbox_blocks_getmetatable(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block getmetatable (B.3 - metatable access)."""
        result = sandbox.execute("return getmetatable({})")

        assert result.success is False
        assert result.error_type == "sandbox"
        assert "getmetatable" in result.error.lower()

    def test_sandbox_blocks_setmetatable(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block setmetatable (B.3 - metatable access)."""
        result = sandbox.execute("setmetatable({}, {})")

        assert result.success is False
        assert result.error_type == "sandbox"
        assert "setmetatable" in result.error.lower()

    def test_sandbox_blocks_rawget(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block rawget."""
        result = sandbox.execute("rawget({}, 'key')")

        assert result.success is False
        assert result.error_type == "sandbox"

    def test_sandbox_blocks_rawset(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block rawset."""
        result = sandbox.execute("rawset({}, 'key', 'value')")

        assert result.success is False
        assert result.error_type == "sandbox"

    def test_sandbox_blocks_debug_module(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block debug module."""
        result = sandbox.execute("debug.getinfo(1)")

        assert result.success is False
        assert result.error_type == "sandbox"

    def test_sandbox_blocks_package_module(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block package module."""
        result = sandbox.execute("package.loadlib('lib', 'func')")

        assert result.success is False
        assert result.error_type == "sandbox"

    def test_sandbox_blocks_load(self, sandbox: LuaSandbox) -> None:
        """Sandbox must block load function."""
        result = sandbox.execute("load('return 1')()")

        assert result.success is False
        assert result.error_type == "sandbox"

    def test_sandbox_allows_safe_globals(self, sandbox: LuaSandbox) -> None:
        """Sandbox should allow safe globals."""
        # These should all work
        safe_operations = [
            ("return type(1)", "number"),
            ("return tostring(42)", "42"),
            ("return tonumber('42')", 42.0),
            ("return #'hello'", 5.0),
            ("local ok, err = pcall(function() error('test') end); return ok", False),
        ]

        for code, expected in safe_operations:
            result = sandbox.execute(code)
            assert result.success is True, f"Failed: {code}"
            assert result.result == expected, f"Wrong result for {code}"


# =============================================================================
# Timeout Tests
# =============================================================================


@requires_lupa
class TestSandboxTimeout:
    """Tests for sandbox timeout enforcement (E5003)."""

    def test_timeout_infinite_loop(self) -> None:
        """Sandbox should timeout on infinite loop."""
        sandbox = LuaSandbox(timeout_seconds=0.5)  # Very short timeout

        result = sandbox.execute("while true do end")

        assert result.success is False
        assert result.error_type == "timeout"

    def test_timeout_long_computation(self) -> None:
        """Sandbox should timeout on long computation."""
        sandbox = LuaSandbox(timeout_seconds=0.1)

        # This should take longer than 0.1 seconds (infinite nested loops)
        result = sandbox.execute("""
            local sum = 0
            for i = 1, 1000000000000 do
                for j = 1, 1000000000000 do
                    sum = sum + i + j
                end
            end
            return sum
        """)

        assert result.success is False
        assert result.error_type == "timeout"

    def test_fast_execution_succeeds(self) -> None:
        """Sandbox should not timeout on fast execution."""
        sandbox = LuaSandbox(timeout_seconds=5.0)

        result = sandbox.execute("""
            local sum = 0
            for i = 1, 100 do
                sum = sum + i
            end
            return sum
        """)

        assert result.success is True
        assert result.result == 5050.0


# =============================================================================
# Error Code Verification
# =============================================================================


class TestErrorCodes:
    """Verify error codes match errors.yaml."""

    def test_error_codes_defined(self) -> None:
        """Verify all expected error codes are defined."""
        assert ERROR_CODES["syntax"] == "E5001"
        assert ERROR_CODES["runtime"] == "E5002"
        assert ERROR_CODES["timeout"] == "E5003"
        assert ERROR_CODES["sandbox"] == "E7001"


# =============================================================================
# Type Conversion Tests
# =============================================================================


@requires_lupa
class TestTypeConversion:
    """Tests for Lua-Python type conversion."""

    def test_number_to_float(self, sandbox: LuaSandbox) -> None:
        """Lua numbers should become Python floats."""
        result = sandbox.execute("return 42")

        assert result.success is True
        assert isinstance(result.result, float)
        assert result.result == 42.0

    def test_string_preserved(self, sandbox: LuaSandbox) -> None:
        """Lua strings should become Python strings."""
        result = sandbox.execute("return 'hello'")

        assert result.success is True
        assert isinstance(result.result, str)
        assert result.result == "hello"

    def test_boolean_preserved(self, sandbox: LuaSandbox) -> None:
        """Lua booleans should become Python booleans."""
        result = sandbox.execute("return true")

        assert result.success is True
        assert isinstance(result.result, bool)
        assert result.result is True

    def test_nil_to_none(self, sandbox: LuaSandbox) -> None:
        """Lua nil should become Python None."""
        result = sandbox.execute("return nil")

        assert result.success is True
        assert result.result is None

    def test_table_to_dict(self, sandbox: LuaSandbox) -> None:
        """Lua dict-like tables should become Python dicts."""
        result = sandbox.execute("return {a = 1, b = 2}")

        assert result.success is True
        assert isinstance(result.result, dict)
        assert result.result["a"] == 1.0
        assert result.result["b"] == 2.0

    def test_array_table_to_list(self, sandbox: LuaSandbox) -> None:
        """Lua array-like tables should become Python lists."""
        result = sandbox.execute("return {1, 2, 3}")

        assert result.success is True
        assert isinstance(result.result, list)
        assert result.result == [1.0, 2.0, 3.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
