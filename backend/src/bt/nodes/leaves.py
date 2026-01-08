"""
Leaf Nodes - Nodes that perform actual work with no children.

This module implements the leaf nodes from contracts/nodes.yaml:
- LeafNode: Base class for all leaf nodes
- Action: Executes a Python function
- Condition: Evaluates a boolean condition
- SubtreeRef: References another tree from registry
- Script: Executes Lua script in sandbox

Error codes handled:
- E3001: Tree not found (SubtreeRef)
- E3002: Circular reference (SubtreeRef)
- E4003: Function not found (Action)
- E5001: Lua syntax error (Script)
- E5002: Lua runtime error (Script)
- E5003: Lua timeout (Script)
- E5004: Lua intended failure (Script)
- E7001: Sandbox violation (Script)

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.3.1-2.3.4, 2.4.4-2.4.6 from tasks.md
"""

from __future__ import annotations

import importlib
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Type,
    Union,
)

from pydantic import BaseModel

from ..state.base import NodeType, RunStatus
from ..state.contracts import NodeContract, get_action_contract
from .base import BehaviorNode

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..core.tree import BehaviorTree
    from ..lua.sandbox import LuaSandbox, LuaExecutionResult
    from ..state.blackboard import TypedBlackboard


logger = logging.getLogger(__name__)


# =============================================================================
# Error Classes
# =============================================================================


class FunctionNotFoundError(Exception):
    """Exception raised when a function path doesn't resolve.

    Error code: E4003 (from errors.yaml)
    """

    def __init__(
        self,
        fn_path: str,
        node_name: str,
        available_modules: Optional[List[str]] = None,
        source_location: Optional[str] = None,
    ) -> None:
        self.fn_path = fn_path
        self.node_name = node_name
        self.available_modules = available_modules or []
        self.source_location = source_location
        self.error_code = "E4003"

        message = (
            f"[E4003] Function '{fn_path}' not found for action '{node_name}'. "
            f"Ensure module is importable and function exists."
        )
        if available_modules:
            message += f" Available modules: {available_modules[:5]}"

        super().__init__(message)


class TreeNotFoundError(Exception):
    """Exception raised when a referenced tree doesn't exist.

    Error code: E3001 (from errors.yaml)
    """

    def __init__(
        self,
        tree_name: str,
        requested_by: str,
        available_trees: Optional[List[str]] = None,
    ) -> None:
        self.tree_name = tree_name
        self.requested_by = requested_by
        self.available_trees = available_trees or []
        self.error_code = "E3001"

        message = f"[E3001] Tree '{tree_name}' not found in registry."
        if available_trees:
            message += f" Available trees: {available_trees[:5]}"

        super().__init__(message)


class CircularReferenceError(Exception):
    """Exception raised when a circular tree reference is detected.

    Error code: E3002 (from errors.yaml)
    """

    def __init__(
        self,
        cycle_path: List[str],
        source_location: Optional[str] = None,
    ) -> None:
        self.cycle_path = cycle_path
        self.source_location = source_location
        self.error_code = "E3002"

        cycle_str = " -> ".join(cycle_path)
        message = f"[E3002] Circular tree reference detected: {cycle_str}"

        super().__init__(message)


# =============================================================================
# LeafNode Base Class
# =============================================================================


class LeafNode(BehaviorNode):
    """Base class for nodes with no children.

    From contracts/nodes.yaml leaf_nodes section:
    - node_type is always LEAF
    - children is always empty list
    - Performs actual work (actions, conditions, scripts)

    All leaf nodes must implement _tick() which contains the
    node-specific logic.
    """

    @property
    def node_type(self) -> NodeType:
        """Leaf nodes have no children."""
        return NodeType.LEAF

    @property
    def children(self) -> List[BehaviorNode]:
        """Leaf nodes always have empty children list."""
        return []

    def _add_child(self, child: BehaviorNode) -> None:
        """Leaf nodes cannot have children."""
        raise ValueError(
            f"Cannot add child to leaf node '{self._id}'. "
            f"Leaf nodes have no children."
        )


# =============================================================================
# Action Node
# =============================================================================


# Type for action functions
ActionFunction = Callable[["TickContext"], RunStatus]


class Action(LeafNode):
    """Executes a Python function.

    From contracts/nodes.yaml Action leaf:
    - Call configured function with blackboard access
    - Function returns RunStatus
    - May return RUNNING for async operations
    - E4003 if function path doesn't resolve

    The function can be:
    1. A dotted path string like 'oracle.load_context' (resolved at build time)
    2. A callable directly passed

    Example:
        >>> action = Action(
        ...     id="load-ctx",
        ...     fn="oracle.actions.load_context",
        ... )

        >>> # Or with direct callable
        >>> action = Action(
        ...     id="load-ctx",
        ...     fn=my_load_context_function,
        ... )
    """

    def __init__(
        self,
        id: str,
        fn: Union[str, ActionFunction],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize an Action node.

        Args:
            id: Unique node identifier.
            fn: Function path (dotted string) or callable.
            name: Human-readable name (defaults to id).
            metadata: Optional metadata for debugging.

        Raises:
            FunctionNotFoundError: If fn is a string path that doesn't resolve.
        """
        super().__init__(id=id, name=name, metadata=metadata)

        self._fn_path: Optional[str] = None
        self._fn: ActionFunction

        if isinstance(fn, str):
            # Resolve function path at init time
            self._fn_path = fn
            self._fn = self._resolve_function(fn)
        elif callable(fn):
            self._fn = fn
        else:
            raise TypeError(
                f"fn must be a string path or callable, got {type(fn)}"
            )

        # Cache the contract from the function if decorated
        self._fn_contract = get_action_contract(self._fn)

    def _resolve_function(self, fn_path: str) -> ActionFunction:
        """Resolve a dotted function path to a callable.

        Args:
            fn_path: Dotted path like 'module.submodule.function'.

        Returns:
            The resolved callable.

        Raises:
            FunctionNotFoundError: If path cannot be resolved.
        """
        try:
            # Split into module path and function name
            parts = fn_path.rsplit(".", 1)
            if len(parts) != 2:
                raise FunctionNotFoundError(
                    fn_path=fn_path,
                    node_name=self._id,
                    source_location=None,
                )

            module_path, func_name = parts

            # Try to import the module
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                raise FunctionNotFoundError(
                    fn_path=fn_path,
                    node_name=self._id,
                    source_location=str(e),
                ) from e

            # Get the function
            if not hasattr(module, func_name):
                available = [
                    name for name in dir(module)
                    if not name.startswith("_") and callable(getattr(module, name))
                ]
                raise FunctionNotFoundError(
                    fn_path=fn_path,
                    node_name=self._id,
                    available_modules=available,
                )

            fn = getattr(module, func_name)

            if not callable(fn):
                raise FunctionNotFoundError(
                    fn_path=fn_path,
                    node_name=self._id,
                    source_location=f"{func_name} is not callable",
                )

            return fn

        except FunctionNotFoundError:
            raise
        except Exception as e:
            raise FunctionNotFoundError(
                fn_path=fn_path,
                node_name=self._id,
                source_location=str(e),
            ) from e

    @classmethod
    def contract(cls) -> NodeContract:
        """Contract is defined per action via @action_contract decorator.

        Returns the base empty contract. Actual contract comes from
        the decorated function if available.
        """
        return NodeContract(description="Action node - contract defined per function")

    def get_effective_contract(self) -> NodeContract:
        """Get the actual contract for this action.

        Returns the function's contract if decorated, otherwise
        the base Action contract.
        """
        if self._fn_contract:
            return self._fn_contract
        return self.__class__.contract()

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute the configured function.

        Args:
            ctx: Tick context with blackboard access.

        Returns:
            RunStatus from the function execution.
        """
        try:
            result = self._fn(ctx)

            # Validate return type
            if not isinstance(result, RunStatus):
                logger.warning(
                    f"Action '{self._id}' returned {type(result).__name__} "
                    f"instead of RunStatus, treating as FAILURE"
                )
                return RunStatus.FAILURE

            return result

        except Exception as e:
            logger.error(
                f"Action '{self._id}' raised exception: {e}",
                exc_info=True,
            )
            return RunStatus.FAILURE

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information including function path."""
        info = super().debug_info()
        info["fn_path"] = self._fn_path
        info["has_function_contract"] = self._fn_contract is not None
        return info


# =============================================================================
# Condition Node
# =============================================================================


# Type for condition functions
ConditionFunction = Callable[["TickContext"], bool]


class Condition(LeafNode):
    """Evaluates a boolean condition.

    From contracts/nodes.yaml Condition leaf:
    - Evaluate condition
    - Return SUCCESS if true, FAILURE if false
    - NEVER returns RUNNING

    The condition can be:
    1. A callable that returns bool
    2. A Lua expression string (evaluated via sandbox)

    Example:
        >>> cond = Condition(
        ...     id="has-budget",
        ...     condition=lambda ctx: ctx.blackboard.get("budget", BudgetModel).value > 0,
        ... )

        >>> # Or with Lua expression
        >>> cond = Condition(
        ...     id="has-budget",
        ...     condition="bb.budget > 0",
        ... )
    """

    def __init__(
        self,
        id: str,
        condition: Union[str, ConditionFunction],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a Condition node.

        Args:
            id: Unique node identifier.
            condition: Lua expression string or Python callable returning bool.
            name: Human-readable name (defaults to id).
            metadata: Optional metadata for debugging.
        """
        super().__init__(id=id, name=name, metadata=metadata)

        self._condition_expr: Optional[str] = None
        self._condition_fn: Optional[ConditionFunction] = None

        if isinstance(condition, str):
            self._condition_expr = condition
        elif callable(condition):
            self._condition_fn = condition
        else:
            raise TypeError(
                f"condition must be a string or callable, got {type(condition)}"
            )

    @classmethod
    def contract(cls) -> NodeContract:
        """Condition contract - inputs are condition dependencies, no outputs."""
        return NodeContract(
            description="Condition node - evaluates to SUCCESS (true) or FAILURE (false)"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Evaluate the condition.

        Returns SUCCESS if true, FAILURE if false.
        Never returns RUNNING.

        Args:
            ctx: Tick context with blackboard access.

        Returns:
            SUCCESS if condition is true, FAILURE otherwise.
        """
        try:
            if self._condition_fn is not None:
                # Python callable
                result = self._condition_fn(ctx)
            elif self._condition_expr is not None:
                # Lua expression - evaluate via sandbox
                result = self._evaluate_lua_condition(ctx)
            else:
                logger.error(f"Condition '{self._id}' has no condition defined")
                return RunStatus.FAILURE

            # Convert bool to RunStatus
            return RunStatus.from_bool(bool(result))

        except Exception as e:
            logger.error(
                f"Condition '{self._id}' raised exception: {e}",
                exc_info=True,
            )
            return RunStatus.FAILURE

    def _evaluate_lua_condition(self, ctx: "TickContext") -> bool:
        """Evaluate Lua expression condition.

        Creates a minimal Lua environment with blackboard access
        and evaluates the condition expression.

        Args:
            ctx: Tick context with blackboard access.

        Returns:
            Boolean result of the condition.
        """
        # Import here to avoid circular imports
        from ..lua.sandbox import LuaSandbox
        from ..state.bridges import LuaStateBridge

        # Create sandbox with short timeout for conditions
        sandbox = LuaSandbox(timeout_seconds=1.0, max_memory_mb=10)

        # Build Lua environment with blackboard
        env = {}
        if ctx.blackboard:
            env["bb"] = LuaStateBridge.to_lua_table(ctx.blackboard)

        # Wrap expression in return statement if needed
        expr = self._condition_expr
        if not expr.strip().startswith("return"):
            expr = f"return {expr}"

        # Execute and get result
        result = sandbox.execute(expr, env=env, source_name=f"condition:{self._id}")

        if not result.success:
            logger.warning(
                f"Condition '{self._id}' Lua evaluation failed: {result.error}"
            )
            return False

        # Convert result to bool
        return bool(result.result)

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information including condition type."""
        info = super().debug_info()
        info["condition_type"] = "lua" if self._condition_expr else "python"
        if self._condition_expr:
            info["condition_expr"] = self._condition_expr
        return info


# =============================================================================
# SubtreeRef Node
# =============================================================================


class SubtreeRef(LeafNode):
    """References another tree as a subtree.

    From contracts/nodes.yaml SubtreeRef leaf:
    - Resolve tree from registry
    - Create child blackboard scope
    - Tick subtree root
    - Merge results back per merge config
    - Support lazy=True for runtime resolution
    - E3001 if tree not found
    - E3002 for circular references

    Example:
        >>> ref = SubtreeRef(
        ...     id="run-research",
        ...     tree_name="research-runner",
        ... )

        >>> # Lazy resolution (runtime)
        >>> ref = SubtreeRef(
        ...     id="dynamic-tree",
        ...     tree_name="computed-tree-name",
        ...     lazy=True,
        ... )
    """

    def __init__(
        self,
        id: str,
        tree_name: str,
        lazy: bool = False,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a SubtreeRef node.

        Args:
            id: Unique node identifier.
            tree_name: Name of tree to reference in registry.
            lazy: If True, resolve at runtime instead of load time.
            name: Human-readable name (defaults to id).
            metadata: Optional metadata for debugging.
        """
        super().__init__(id=id, name=name, metadata=metadata)

        self._tree_name = tree_name
        self._lazy = lazy
        self._resolved_tree: Optional["BehaviorTree"] = None

        # Track reference chain for circular detection
        self._reference_chain: List[str] = []

    @classmethod
    def contract(cls) -> NodeContract:
        """SubtreeRef inherits contract from referenced tree."""
        return NodeContract(
            description="SubtreeRef - references another tree"
        )

    def resolve(
        self,
        registry: Dict[str, "BehaviorTree"],
        reference_chain: Optional[List[str]] = None,
    ) -> None:
        """Resolve the tree reference from registry.

        Called at build time for eager resolution, or at first tick for lazy.

        Args:
            registry: Dictionary mapping tree names to BehaviorTree instances.
            reference_chain: Current chain for circular detection.

        Raises:
            TreeNotFoundError: If tree not in registry.
            CircularReferenceError: If circular reference detected.
        """
        # Track chain for circular detection
        chain = reference_chain or []
        self._reference_chain = chain

        # Check for circular reference
        if self._tree_name in chain:
            cycle = chain + [self._tree_name]
            raise CircularReferenceError(cycle_path=cycle)

        # Look up tree
        if self._tree_name not in registry:
            raise TreeNotFoundError(
                tree_name=self._tree_name,
                requested_by=self._id,
                available_trees=list(registry.keys()),
            )

        self._resolved_tree = registry[self._tree_name]

        # Recursively resolve subtree's SubtreeRefs
        new_chain = chain + [self._tree_name]
        self._resolve_nested_refs(self._resolved_tree.root, registry, new_chain)

    def _resolve_nested_refs(
        self,
        node: BehaviorNode,
        registry: Dict[str, "BehaviorTree"],
        chain: List[str],
    ) -> None:
        """Recursively resolve SubtreeRef nodes in a tree."""
        if isinstance(node, SubtreeRef) and node._resolved_tree is None:
            node.resolve(registry, chain)

        # Check children
        for child in node.children:
            self._resolve_nested_refs(child, registry, chain)

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Tick the referenced subtree.

        Creates a child blackboard scope, ticks the subtree root,
        and returns the result.

        Args:
            ctx: Tick context with blackboard access.

        Returns:
            RunStatus from subtree execution.
        """
        # Lazy resolution if needed
        if self._resolved_tree is None:
            if not self._lazy:
                logger.error(
                    f"SubtreeRef '{self._id}' not resolved and lazy=False. "
                    f"Call resolve() before ticking."
                )
                return RunStatus.FAILURE

            # Try to resolve from services
            if ctx.services and hasattr(ctx.services, "tree_registry"):
                try:
                    self.resolve(ctx.services.tree_registry, self._reference_chain)
                except (TreeNotFoundError, CircularReferenceError) as e:
                    logger.error(f"SubtreeRef '{self._id}' resolution failed: {e}")
                    return RunStatus.FAILURE
            else:
                logger.error(
                    f"SubtreeRef '{self._id}' is lazy but no tree_registry in services"
                )
                return RunStatus.FAILURE

        # Create child scope for subtree
        if ctx.blackboard:
            child_scope = ctx.blackboard.create_child_scope(
                f"subtree:{self._tree_name}"
            )
            child_ctx = ctx.with_blackboard(child_scope)
        else:
            child_ctx = ctx

        # Tick subtree root
        try:
            result = self._resolved_tree.root.tick(child_ctx)
            return result

        except Exception as e:
            logger.error(
                f"SubtreeRef '{self._id}' failed: {e}",
                exc_info=True,
            )
            return RunStatus.FAILURE

    def reset(self) -> None:
        """Reset subtree ref and the referenced tree."""
        super().reset()

        # Reset the subtree if resolved
        if self._resolved_tree is not None:
            self._resolved_tree.root.reset()

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information including tree reference."""
        info = super().debug_info()
        info["tree_name"] = self._tree_name
        info["lazy"] = self._lazy
        info["resolved"] = self._resolved_tree is not None
        if self._resolved_tree:
            info["resolved_tree_id"] = self._resolved_tree.id
        return info


# =============================================================================
# Script Node
# =============================================================================


class Script(LeafNode):
    """Executes Lua script in sandbox.

    From contracts/nodes.yaml Script leaf:
    - Execute Lua code in sandbox
    - Script accesses blackboard via ctx.bb
    - Script returns {status='success'|'failure'|'running', ...}
    - Parse return value, update blackboard, return status
    - Support inline code or file reference
    - Contract declared via BT.contract() in Lua

    Error codes:
    - E5001: Lua syntax error
    - E5002: Lua runtime error
    - E5003: Script timeout
    - E5004: Lua intended failure (warning only)
    - E7001: Sandbox violation

    Example:
        >>> script = Script(
        ...     id="compute-result",
        ...     code='''
        ...         local x = bb.get("input")
        ...         bb.set("output", x * 2)
        ...         return {status = "success"}
        ...     ''',
        ... )

        >>> # Or from file
        >>> script = Script(
        ...     id="run-script",
        ...     file="scripts/my_script.lua",
        ... )
    """

    # Default timeout from nodes.yaml
    DEFAULT_TIMEOUT_MS = 5000  # 5 seconds
    MAX_TIMEOUT_MS = 30000  # 30 seconds

    def __init__(
        self,
        id: str,
        code: Optional[str] = None,
        file: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a Script node.

        Args:
            id: Unique node identifier.
            code: Inline Lua code (mutually exclusive with file).
            file: Path to Lua file (mutually exclusive with code).
            timeout_ms: Script timeout in milliseconds.
            name: Human-readable name (defaults to id).
            metadata: Optional metadata for debugging.

        Raises:
            ValueError: If neither code nor file is provided, or both are.
        """
        super().__init__(id=id, name=name, metadata=metadata)

        if code is None and file is None:
            raise ValueError("Either 'code' or 'file' must be provided")
        if code is not None and file is not None:
            raise ValueError("Cannot specify both 'code' and 'file'")

        self._code = code
        self._file = file
        self._timeout_ms = min(
            timeout_ms or self.DEFAULT_TIMEOUT_MS,
            self.MAX_TIMEOUT_MS,
        )

        # Cached file content
        self._file_content: Optional[str] = None

        # Sandbox instance (created on first tick)
        self._sandbox: Optional["LuaSandbox"] = None

    @classmethod
    def contract(cls) -> NodeContract:
        """Script contract is declared in Lua via BT.contract()."""
        return NodeContract(
            description="Script node - contract declared in Lua"
        )

    def _get_code(self) -> str:
        """Get the Lua code to execute.

        Returns:
            The Lua code string.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if self._code is not None:
            return self._code

        if self._file_content is not None:
            return self._file_content

        # Load from file
        if self._file is not None:
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    self._file_content = f.read()
                return self._file_content
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Script file not found: {self._file}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read script file '{self._file}': {e}"
                ) from e

        raise ValueError("No code or file specified")

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute the Lua script.

        Args:
            ctx: Tick context with blackboard access.

        Returns:
            RunStatus based on script return value.
        """
        from ..lua.sandbox import LuaSandbox
        from ..state.bridges import LuaStateBridge

        # Create sandbox on first use
        if self._sandbox is None:
            self._sandbox = LuaSandbox(
                timeout_seconds=self._timeout_ms / 1000.0,
                max_memory_mb=50,
            )

        # Get code
        try:
            code = self._get_code()
        except Exception as e:
            logger.error(f"Script '{self._id}' failed to get code: {e}")
            return RunStatus.FAILURE

        # Build environment with blackboard access
        env: Dict[str, Any] = {}
        if ctx.blackboard:
            env["bb"] = LuaStateBridge.to_lua_table(ctx.blackboard)

            # Also provide bb.get and bb.set functions
            # These will be proper Lua wrappers in the sandbox
            env["_bb_ref"] = ctx.blackboard

        # Execute script
        source_name = self._file or f"<script:{self._id}>"
        result = self._sandbox.execute(
            code,
            env=env,
            source_name=source_name,
        )

        # Handle execution result
        return self._handle_result(result, ctx)

    def _handle_result(
        self,
        result: "LuaExecutionResult",
        ctx: "TickContext",
    ) -> RunStatus:
        """Handle Lua execution result.

        Args:
            result: Execution result from sandbox.
            ctx: Tick context.

        Returns:
            Appropriate RunStatus.
        """
        from ..state.bridges import LuaStateBridge

        if not result.success:
            # Map error type to error code
            error_code = {
                "syntax": "E5001",
                "runtime": "E5002",
                "timeout": "E5003",
                "sandbox": "E7001",
            }.get(result.error_type, "E5002")

            logger.error(
                f"[{error_code}] Script '{self._id}' failed: {result.error} "
                f"(line {result.line_number or '?'})"
            )
            return RunStatus.FAILURE

        # Parse return value
        status_str = LuaStateBridge.extract_status_from_lua(result.result)

        if status_str is None:
            # No status returned - default to success
            return RunStatus.SUCCESS

        if status_str == "success":
            # Update blackboard from return value if applicable
            self._update_blackboard_from_result(result.result, ctx)
            return RunStatus.SUCCESS

        elif status_str == "running":
            return RunStatus.RUNNING

        elif status_str == "failure":
            # E5004: Intended failure (logged as warning, not error)
            reason = LuaStateBridge.extract_error_from_lua(result.result)
            logger.warning(
                f"[E5004] Script '{self._id}' returned failure: {reason or 'unspecified'}"
            )
            return RunStatus.FAILURE

        else:
            logger.warning(
                f"Script '{self._id}' returned unknown status: {status_str}"
            )
            return RunStatus.FAILURE

    def _update_blackboard_from_result(
        self,
        lua_result: Any,
        ctx: "TickContext",
    ) -> None:
        """Update blackboard from Lua return value.

        Extracts values from the Lua return table and writes them
        to the blackboard.

        Args:
            lua_result: The Lua return value (typically a table).
            ctx: Tick context with blackboard.
        """
        from ..state.blackboard import lua_to_python

        if ctx.blackboard is None:
            return

        try:
            py_result = lua_to_python(lua_result)

            if not isinstance(py_result, dict):
                return

            # Write any keys that aren't 'status' or 'reason'/'error'/'message'
            skip_keys = {"status", "reason", "error", "message"}
            for key, value in py_result.items():
                if key in skip_keys:
                    continue

                # Only write if key is registered
                if ctx.blackboard._get_registered_schema(key):
                    schema = ctx.blackboard._get_registered_schema(key)
                    try:
                        validated = schema.model_validate(value)
                        result = ctx.blackboard.set(key, validated)
                        if result.is_error:
                            logger.warning(
                                f"Script '{self._id}' failed to set '{key}': {result.error}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Script '{self._id}' failed to validate '{key}': {e}"
                        )

        except Exception as e:
            logger.warning(
                f"Script '{self._id}' failed to process result: {e}"
            )

    def reset(self) -> None:
        """Reset script node state."""
        super().reset()
        # Keep sandbox for reuse

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information."""
        info = super().debug_info()
        info["source_type"] = "file" if self._file else "inline"
        info["file"] = self._file
        info["timeout_ms"] = self._timeout_ms
        info["code_length"] = len(self._code) if self._code else 0
        return info


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Base class
    "LeafNode",
    # Leaf node types
    "Action",
    "Condition",
    "SubtreeRef",
    "Script",
    # Error types
    "FunctionNotFoundError",
    "TreeNotFoundError",
    "CircularReferenceError",
    # Type aliases
    "ActionFunction",
    "ConditionFunction",
]
