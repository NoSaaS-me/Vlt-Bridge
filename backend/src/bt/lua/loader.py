"""
TreeLoader - Loads behavior tree definitions from Lua DSL files.

This module provides the TreeLoader class that:
1. Reads Lua tree definition files
2. Creates a LuaSandbox with injected BT.* API
3. Executes the Lua code to produce TreeDefinition
4. Returns the definition for building into executable trees

From tree-loader.yaml TreeLoader interface:
- load(path: Path) -> TreeDefinition
- load_string(lua_code: str, source_name: str) -> TreeDefinition
- _inject_bt_api(env: Dict[str, Any]) -> None

Error codes:
- E4001: File not found
- E4002: Invalid structure (no BT.tree() call)
- E5001: Lua syntax error
- E5003: Script timeout

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.6.1-2.6.6 from tasks.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .api import BTApiBuilder, NodeDefinition, TreeDefinition, compute_source_hash
from .sandbox import LuaSandbox, LuaExecutionResult

logger = logging.getLogger(__name__)


# =============================================================================
# Error Classes
# =============================================================================


class TreeLoadError(Exception):
    """Exception raised when tree loading fails.

    Error codes from errors.yaml:
    - E4001: File not found
    - E4002: Invalid structure (no BT.tree() call)
    - E5001: Lua syntax error
    - E5003: Script timeout
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        source_path: Optional[str] = None,
        line_number: Optional[int] = None,
    ) -> None:
        self.error_code = error_code
        self.source_path = source_path
        self.line_number = line_number

        location = ""
        if source_path:
            location = f" in '{source_path}'"
            if line_number:
                location += f" at line {line_number}"

        super().__init__(f"[{error_code}]{location}: {message}")


# =============================================================================
# TreeLoader
# =============================================================================


class TreeLoader:
    """Loads behavior tree definitions from Lua DSL files.

    Provides a sandboxed environment for executing Lua tree definitions
    with the BT.* API injected. Returns TreeDefinition objects that can
    be built into executable BehaviorTree instances.

    From tree-loader.yaml:
    - load(path: Path) -> TreeDefinition
    - load_string(lua_code: str, source_name: str) -> TreeDefinition
    - _inject_bt_api(env: Dict[str, Any]) -> None

    Example:
        >>> loader = TreeLoader()
        >>> tree_def = loader.load(Path("trees/oracle-agent.lua"))
        >>> tree_def.name
        'oracle-agent'

        >>> tree_def = loader.load_string('''
        ...     return BT.tree("test", {
        ...         root = BT.sequence({
        ...             BT.action("step1", {fn = "test.step1"}),
        ...         })
        ...     })
        ... ''')
    """

    # Default timeout for Lua execution
    DEFAULT_TIMEOUT_SECONDS = 5.0

    def __init__(self, sandbox_timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        """Initialize the tree loader.

        Args:
            sandbox_timeout: Timeout for Lua execution in seconds.

        Raises:
            ImportError: If lupa is not available.
        """
        self._sandbox_timeout = sandbox_timeout

    def load(self, path: Path) -> TreeDefinition:
        """Load tree definition from Lua file.

        Args:
            path: Path to the .lua file.

        Returns:
            TreeDefinition parsed from the file.

        Raises:
            TreeLoadError: With code:
                - E4001: File not found
                - E5001: Lua syntax error
                - E5003: Script timeout
                - E4002: Invalid structure (no BT.tree() call)
        """
        # Validate path exists
        if not path.exists():
            raise TreeLoadError(
                error_code="E4001",
                message=f"File not found: {path}",
                source_path=str(path),
            )

        # Validate extension
        if path.suffix.lower() != ".lua":
            raise TreeLoadError(
                error_code="E4001",
                message=f"Expected .lua file, got: {path.suffix}",
                source_path=str(path),
            )

        # Read file content
        try:
            content = path.read_text(encoding="utf-8")
        except PermissionError as e:
            raise TreeLoadError(
                error_code="E4001",
                message=f"Permission denied reading file: {e}",
                source_path=str(path),
            )
        except OSError as e:
            raise TreeLoadError(
                error_code="E4001",
                message=f"Error reading file: {e}",
                source_path=str(path),
            )

        # Load from string
        tree_def = self.load_string(content, source_name=str(path))

        # Set source path and hash
        tree_def.source_path = str(path)
        tree_def.source_hash = compute_source_hash(content)

        return tree_def

    def load_string(
        self,
        lua_code: str,
        source_name: str = "<string>",
    ) -> TreeDefinition:
        """Load tree from Lua string (for testing).

        Args:
            lua_code: Lua code defining the tree.
            source_name: Name for error reporting.

        Returns:
            TreeDefinition parsed from the code.

        Raises:
            TreeLoadError: With code:
                - E5001: Lua syntax error
                - E5003: Script timeout
                - E4002: Invalid structure (no BT.tree() call)
        """
        # Create sandbox
        sandbox = LuaSandbox(
            timeout_seconds=self._sandbox_timeout,
            max_memory_mb=50,
        )

        # Build BT.* API
        api_builder = BTApiBuilder(source_path=source_name)
        bt_api = api_builder.build_api()

        # Build environment with BT namespace
        env: Dict[str, Any] = {
            "BT": bt_api,
        }

        # Inject API
        self._inject_bt_api(sandbox, env)

        # Execute Lua code
        result = sandbox.execute(lua_code, env=env, source_name=source_name)

        # Handle execution errors
        if not result.success:
            self._raise_execution_error(result, source_name)

        # Extract TreeDefinition from result
        tree_def = self._extract_tree_definition(result.result, source_name)

        # Set source info
        tree_def.source_path = source_name
        tree_def.source_hash = compute_source_hash(lua_code)

        return tree_def

    def _inject_bt_api(self, sandbox: LuaSandbox, env: Dict[str, Any]) -> None:
        """Inject BT.* functions into sandbox environment.

        Creates the BT namespace table in the Lua environment with all
        tree definition functions available.

        From tree-loader.yaml _inject_bt_api:
        - BT.tree, BT.sequence, BT.selector, BT.parallel
        - BT.action, BT.condition, BT.llm_call, BT.subtree_ref
        - BT.for_each, BT.script
        - BT.timeout, BT.retry, BT.guard, BT.cooldown
        - BT.inverter, BT.always_succeed, BT.always_fail
        - BT.contract
        - BT.tool, BT.oracle, BT.code_search, BT.vault_search

        Args:
            sandbox: LuaSandbox instance.
            env: Environment dictionary to modify.
        """
        # The BT namespace is already built in env by the caller
        # This method can be extended to add sandbox-specific wrappers

        # Optionally add line tracking for better error messages
        # This would require sandbox support for debug.getinfo

        logger.debug(f"Injected BT.* API with {len(env.get('BT', {}))} functions")

    def _raise_execution_error(
        self,
        result: LuaExecutionResult,
        source_name: str,
    ) -> None:
        """Raise appropriate TreeLoadError for execution failure.

        Args:
            result: Failed LuaExecutionResult.
            source_name: Source name for error context.

        Raises:
            TreeLoadError: With appropriate error code.
        """
        error_code_map = {
            "syntax": "E5001",
            "timeout": "E5003",
            "runtime": "E5002",
            "sandbox": "E7001",
        }

        error_code = error_code_map.get(result.error_type or "", "E5002")

        raise TreeLoadError(
            error_code=error_code,
            message=result.error or "Unknown execution error",
            source_path=source_name,
            line_number=result.line_number,
        )

    def _extract_tree_definition(
        self,
        result: Any,
        source_name: str,
    ) -> TreeDefinition:
        """Extract TreeDefinition from Lua execution result.

        The Lua code should return a TreeDefinition from BT.tree().

        Args:
            result: Return value from Lua execution.
            source_name: Source name for error context.

        Returns:
            TreeDefinition extracted from result.

        Raises:
            TreeLoadError: E4002 if result is not a valid TreeDefinition.
        """
        # Check for TreeDefinition directly
        if isinstance(result, TreeDefinition):
            return result

        # Check for None result
        if result is None:
            raise TreeLoadError(
                error_code="E4002",
                message=(
                    "Lua script did not return a tree definition. "
                    "Make sure to use 'return BT.tree(name, config)'"
                ),
                source_path=source_name,
            )

        # Check for NodeDefinition (user forgot to wrap in BT.tree)
        if isinstance(result, NodeDefinition):
            raise TreeLoadError(
                error_code="E4002",
                message=(
                    f"Lua script returned a NodeDefinition ({result.type}) instead of TreeDefinition. "
                    "Wrap the root node with BT.tree(name, {root = ...})"
                ),
                source_path=source_name,
            )

        # Unknown type
        raise TreeLoadError(
            error_code="E4002",
            message=(
                f"Lua script returned unexpected type: {type(result).__name__}. "
                "Expected TreeDefinition from BT.tree()"
            ),
            source_path=source_name,
        )


# =============================================================================
# Validation Functions
# =============================================================================


def validate_tree_name(name: str) -> bool:
    """Validate tree name matches allowed pattern.

    From tree-loader.yaml file_validation:
    Pattern: ^[a-zA-Z][a-zA-Z0-9_-]*$
    - Start with letter
    - Contain only letters, numbers, underscore, hyphen
    - No dots, slashes, or special characters

    Args:
        name: Tree name to validate.

    Returns:
        True if name is valid.
    """
    import re
    pattern = r"^[a-zA-Z][a-zA-Z0-9_-]*$"
    return bool(re.match(pattern, name))


def validate_node_id(node_id: str) -> bool:
    """Validate node ID matches allowed pattern.

    Same pattern as tree names.

    Args:
        node_id: Node ID to validate.

    Returns:
        True if ID is valid.
    """
    return validate_tree_name(node_id)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "TreeLoader",
    "TreeLoadError",
    "validate_tree_name",
    "validate_node_id",
]
