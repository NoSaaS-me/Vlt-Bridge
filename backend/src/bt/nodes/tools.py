"""
MCP Tool Leaf Nodes - Integrate MCP tools as behavior tree leaf nodes.

This module implements tool integration nodes from contracts/nodes.yaml:
- Tool: Generic MCP tool execution wrapper
- Oracle: Multi-source oracle query with streaming
- CodeSearch: Code search operations via CodeRAG
- VaultSearch: Vault note search via BM25

Error codes handled:
- E4003: Tool not found in registry
- E6001: Tool/async operation timeout
- E2001: Missing required tool parameter

Part of the BT Universal Runtime (spec 019).
Tasks covered: 4.1.1-4.1.5, 4.2.1-4.2.3, 4.3.1-4.3.8 from tasks.md
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

from pydantic import BaseModel

from ..state.base import NodeType, RunStatus, BTError, ErrorContext, RecoveryInfo, RecoveryAction, Severity
from ..state.contracts import NodeContract
from .base import BehaviorNode
from .leaves import LeafNode

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard


logger = logging.getLogger(__name__)


# =============================================================================
# Error Classes
# =============================================================================


class ToolNotFoundError(Exception):
    """Exception raised when a tool is not found in registry.

    Error code: E4003 (from errors.yaml)
    """

    def __init__(
        self,
        tool_name: str,
        node_id: str,
        available_tools: Optional[List[str]] = None,
    ) -> None:
        self.tool_name = tool_name
        self.node_id = node_id
        self.available_tools = available_tools or []
        self.error_code = "E4003"

        message = (
            f"[E4003] Tool '{tool_name}' not found in registry for node '{node_id}'."
        )
        if available_tools:
            message += f" Available tools: {available_tools[:10]}"

        super().__init__(message)


class ToolTimeoutError(Exception):
    """Exception raised when a tool execution times out.

    Error code: E6001 (from errors.yaml)
    """

    def __init__(
        self,
        tool_name: str,
        node_id: str,
        timeout_ms: int,
        operation_id: Optional[str] = None,
    ) -> None:
        self.tool_name = tool_name
        self.node_id = node_id
        self.timeout_ms = timeout_ms
        self.operation_id = operation_id
        self.error_code = "E6001"

        message = (
            f"[E6001] Tool '{tool_name}' timed out after {timeout_ms}ms "
            f"for node '{node_id}'."
        )
        if operation_id:
            message += f" Operation ID: {operation_id}"

        super().__init__(message)


class MissingToolParameterError(Exception):
    """Exception raised when a required tool parameter is missing.

    Error code: E2001 (from errors.yaml)
    """

    def __init__(
        self,
        tool_name: str,
        node_id: str,
        param_name: str,
        expected_type: str,
    ) -> None:
        self.tool_name = tool_name
        self.node_id = node_id
        self.param_name = param_name
        self.expected_type = expected_type
        self.error_code = "E2001"

        message = (
            f"[E2001] Missing required parameter '{param_name}' (type: {expected_type}) "
            f"for tool '{tool_name}' in node '{node_id}'."
        )

        super().__init__(message)


# =============================================================================
# Parameter Interpolation
# =============================================================================

# Pattern for ${bb.key} and ${bb.key | default:value}
BB_INTERPOLATION_PATTERN = re.compile(
    r'\$\{bb\.([a-zA-Z_][a-zA-Z0-9_\.]*)'  # ${bb.key.nested
    r'(?:\s*\|\s*default:([^}]*))?\}'      # | default:value}
)


def interpolate_params(
    params: Dict[str, Any],
    bb: "TypedBlackboard",
) -> Dict[str, Any]:
    """Interpolate ${bb.key} and ${bb.key | default:value} patterns in params.

    From contracts/integrations.yaml interpolation specification:
    - ${bb.key}: Read value from blackboard
    - ${bb.key.nested}: Read nested value (dot notation)
    - ${bb.key | default:value}: Default if key missing

    Args:
        params: Parameter dictionary with potential interpolation patterns.
        bb: TypedBlackboard to read values from.

    Returns:
        New dictionary with interpolated values.
    """
    result = {}

    for key, value in params.items():
        result[key] = _interpolate_value(value, bb)

    return result


def _interpolate_value(value: Any, bb: "TypedBlackboard") -> Any:
    """Recursively interpolate a single value."""
    if isinstance(value, str):
        return _interpolate_string(value, bb)
    elif isinstance(value, dict):
        return {k: _interpolate_value(v, bb) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_value(v, bb) for v in value]
    else:
        return value


def _interpolate_string(s: str, bb: "TypedBlackboard") -> Any:
    """Interpolate a single string value.

    If the entire string is a pattern like ${bb.key}, return the raw value.
    If the pattern is embedded, substitute as string.
    """
    # Check if entire string is a single pattern
    if s.startswith("${bb.") and s.endswith("}") and s.count("${") == 1:
        match = BB_INTERPOLATION_PATTERN.fullmatch(s)
        if match:
            bb_key = match.group(1)
            default_str = match.group(2)

            value = _get_bb_value(bb, bb_key)
            if value is not None:
                return value
            elif default_str is not None:
                return _parse_default(default_str.strip())
            else:
                return None

    # Handle embedded patterns (substitute as strings)
    def replacer(match: re.Match) -> str:
        bb_key = match.group(1)
        default_str = match.group(2)

        value = _get_bb_value(bb, bb_key)
        if value is not None:
            return str(value)
        elif default_str is not None:
            return default_str.strip()
        else:
            return ""

    return BB_INTERPOLATION_PATTERN.sub(replacer, s)


def _get_bb_value(bb: "TypedBlackboard", key: str) -> Any:
    """Get a value from blackboard, supporting dot notation for nested access.

    Args:
        bb: TypedBlackboard instance.
        key: Key with potential dot notation (e.g., "context.session_id").

    Returns:
        Value if found, None otherwise.
    """
    # Split on dots for nested access
    parts = key.split(".")

    # First part is the main key
    main_key = parts[0]

    # Try to get the value without type checking (raw access)
    value = bb._lookup(main_key)
    if value is None:
        return None

    # Navigate nested structure
    for part in parts[1:]:
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, BaseModel):
            value = getattr(value, part, None)
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            return None

        if value is None:
            return None

    return value


def _parse_default(default_str: str) -> Any:
    """Parse a default value string into appropriate Python type."""
    # Try numeric parsing
    try:
        if "." in default_str:
            return float(default_str)
        return int(default_str)
    except ValueError:
        pass

    # Boolean
    if default_str.lower() == "true":
        return True
    if default_str.lower() == "false":
        return False

    # Null
    if default_str.lower() in ("null", "none"):
        return None

    # String (strip quotes if present)
    if (default_str.startswith('"') and default_str.endswith('"')) or \
       (default_str.startswith("'") and default_str.endswith("'")):
        return default_str[1:-1]

    return default_str


# =============================================================================
# Tool Result Model
# =============================================================================


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    value: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


# =============================================================================
# Base Tool Leaf
# =============================================================================


class ToolLeaf(LeafNode):
    """Base class for tool-related leaf nodes.

    Provides common functionality:
    - Request ID tracking for async operations
    - Tool start time tracking
    - Parameter interpolation
    - Timeout handling
    """

    # Default timeout for tools (from nodes.yaml)
    DEFAULT_TIMEOUT_MS = 30000  # 30 seconds

    def __init__(
        self,
        id: str,
        timeout_ms: Optional[int] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize tool leaf base.

        Args:
            id: Unique node identifier.
            timeout_ms: Tool execution timeout in milliseconds.
            name: Human-readable name (defaults to id).
            metadata: Optional metadata for debugging.
        """
        super().__init__(id=id, name=name, metadata=metadata)

        self._timeout_ms = timeout_ms or self.DEFAULT_TIMEOUT_MS

        # State tracking (from nodes.yaml state section)
        self._request_id: Optional[str] = None
        self._tool_start_time: Optional[datetime] = None

    def _check_timeout(self) -> bool:
        """Check if the tool has timed out.

        Returns:
            True if timeout exceeded, False otherwise.
        """
        if self._tool_start_time is None:
            return False

        elapsed = datetime.now(timezone.utc) - self._tool_start_time
        elapsed_ms = elapsed.total_seconds() * 1000
        return elapsed_ms >= self._timeout_ms

    def _get_elapsed_ms(self) -> float:
        """Get elapsed time since tool start.

        Returns:
            Elapsed milliseconds, or 0 if not started.
        """
        if self._tool_start_time is None:
            return 0.0

        elapsed = datetime.now(timezone.utc) - self._tool_start_time
        return elapsed.total_seconds() * 1000

    def reset(self) -> None:
        """Reset tool state for new execution."""
        super().reset()
        self._request_id = None
        self._tool_start_time = None

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information."""
        info = super().debug_info()
        info["timeout_ms"] = self._timeout_ms
        info["request_id"] = self._request_id
        info["tool_start_time"] = (
            self._tool_start_time.isoformat() if self._tool_start_time else None
        )
        info["elapsed_ms"] = self._get_elapsed_ms()
        return info


# =============================================================================
# Tool Node (Generic MCP Tool Wrapper)
# =============================================================================


class Tool(ToolLeaf):
    """Generic MCP tool execution as leaf node.

    From contracts/nodes.yaml Tool leaf:
    - Look up tool in ToolRegistry
    - Interpolate parameters from blackboard (${bb.key} syntax)
    - For sync tools: execute, write result, return SUCCESS/FAILURE
    - For async tools: return RUNNING, track request_id, check completion

    Example:
        >>> tool = Tool(
        ...     id="search-vault",
        ...     tool_name="search_notes",
        ...     params={"query": "${bb.user_query}", "limit": 10},
        ...     output="search_results",
        ... )
    """

    def __init__(
        self,
        id: str,
        tool_name: str,
        params: Dict[str, Any],
        output: str,
        timeout_ms: Optional[int] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a Tool node.

        Args:
            id: Unique node identifier.
            tool_name: MCP tool name from registry.
            params: Tool parameters (supports ${bb.key} interpolation).
            output: Blackboard key for result.
            timeout_ms: Override default timeout.
            name: Human-readable name (defaults to id).
            metadata: Optional metadata for debugging.
        """
        super().__init__(
            id=id,
            timeout_ms=timeout_ms,
            name=name,
            metadata=metadata,
        )

        self._tool_name = tool_name
        self._params = params
        self._output_key = output

    @classmethod
    def contract(cls) -> NodeContract:
        """Tool contract - dynamic based on tool.

        Note: Returns empty contract since actual keys are instance-specific
        and tool outputs are dynamic/unknown at class definition time.
        Use _output_key instance attribute for the actual output key.
        """
        return NodeContract(
            description="Tool node - executes MCP tool from registry"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute the tool.

        Args:
            ctx: Tick context with blackboard and services access.

        Returns:
            SUCCESS if tool executed successfully.
            FAILURE if tool failed or not found.
            RUNNING if async tool is still executing.
        """
        # First tick: start execution
        if self._tool_start_time is None:
            return self._start_execution(ctx)

        # Subsequent ticks: check async completion or timeout
        return self._check_completion(ctx)

    def _start_execution(self, ctx: "TickContext") -> RunStatus:
        """Start tool execution (first tick).

        Args:
            ctx: Tick context.

        Returns:
            RunStatus based on execution mode (sync/async).
        """
        self._tool_start_time = datetime.now(timezone.utc)

        # Get tool executor from services
        tool_executor = self._get_tool_executor(ctx)
        if tool_executor is None:
            logger.error(f"No tool executor available for node '{self._id}'")
            self._write_error(ctx, "Tool executor not available")
            return RunStatus.FAILURE

        # Interpolate parameters
        if ctx.blackboard:
            try:
                interpolated_params = interpolate_params(self._params, ctx.blackboard)
            except Exception as e:
                logger.error(f"Parameter interpolation failed for '{self._id}': {e}")
                self._write_error(ctx, f"Parameter interpolation failed: {e}")
                return RunStatus.FAILURE
        else:
            interpolated_params = self._params

        # Check for required parameters (tool-specific validation)
        missing = self._validate_params(interpolated_params)
        if missing:
            logger.error(
                f"Missing required parameters for tool '{self._tool_name}': {missing}"
            )
            self._write_error(
                ctx,
                f"Missing required parameters: {missing}",
            )
            return RunStatus.FAILURE

        # Execute tool
        try:
            result = tool_executor.execute(
                self._tool_name,
                interpolated_params,
                ctx,
            )

            # Check if this is a sync result or async indicator
            if isinstance(result, str) and result.startswith("async:"):
                # Async tool - track request ID
                self._request_id = result[6:]  # Remove "async:" prefix
                ctx.add_async(self._request_id)
                logger.debug(
                    f"Tool '{self._tool_name}' started async, request_id: {self._request_id}"
                )
                return RunStatus.RUNNING

            # Sync tool - write result
            return self._handle_result(ctx, result)

        except ToolNotFoundError as e:
            logger.error(str(e))
            self._write_error(ctx, str(e))
            return RunStatus.FAILURE

        except Exception as e:
            logger.error(f"Tool '{self._tool_name}' execution failed: {e}")
            self._write_error(ctx, str(e))
            return RunStatus.FAILURE

    def _check_completion(self, ctx: "TickContext") -> RunStatus:
        """Check async tool completion (subsequent ticks).

        Args:
            ctx: Tick context.

        Returns:
            RUNNING if still executing, SUCCESS/FAILURE when done.
        """
        # Check timeout
        if self._check_timeout():
            logger.error(
                f"Tool '{self._tool_name}' timed out after {self._timeout_ms}ms"
            )
            self._write_error(ctx, f"Timeout after {self._timeout_ms}ms")

            # Cancel async operation
            if self._request_id:
                ctx.complete_async(self._request_id)
                tool_executor = self._get_tool_executor(ctx)
                if tool_executor and hasattr(tool_executor, "cancel"):
                    tool_executor.cancel(self._request_id)

            return RunStatus.FAILURE

        # Check completion
        if self._request_id:
            tool_executor = self._get_tool_executor(ctx)
            if tool_executor and hasattr(tool_executor, "check_completion"):
                result = tool_executor.check_completion(self._request_id)
                if result is not None:
                    # Complete - process result
                    ctx.complete_async(self._request_id)
                    return self._handle_result(ctx, result)

        # Still running
        return RunStatus.RUNNING

    def _handle_result(self, ctx: "TickContext", result: Any) -> RunStatus:
        """Handle tool result.

        Args:
            ctx: Tick context.
            result: Tool execution result.

        Returns:
            SUCCESS or FAILURE based on result.
        """
        duration_ms = self._get_elapsed_ms()

        # Write duration
        if ctx.blackboard:
            ctx.blackboard.set_internal("_tool_duration_ms", duration_ms)

        # Check for error result
        if isinstance(result, dict) and result.get("error"):
            error_msg = result.get("message", "Unknown error")
            self._write_error(ctx, error_msg)
            logger.warning(
                f"Tool '{self._tool_name}' returned error: {error_msg}"
            )
            return RunStatus.FAILURE

        # Success - write output
        if ctx.blackboard:
            # Try to write to output key
            # Note: For generic tools, we write as internal since schema may not be registered
            try:
                ctx.blackboard.set_internal(f"_{self._output_key}", result)
            except Exception as e:
                logger.warning(
                    f"Failed to write tool output to blackboard: {e}"
                )

        logger.debug(
            f"Tool '{self._tool_name}' completed in {duration_ms:.2f}ms"
        )
        return RunStatus.SUCCESS

    def _write_error(self, ctx: "TickContext", error: str) -> None:
        """Write error to blackboard."""
        if ctx.blackboard:
            ctx.blackboard.set_internal("_tool_error", error)
            ctx.blackboard.set_internal("_tool_duration_ms", self._get_elapsed_ms())

    def _get_tool_executor(self, ctx: "TickContext") -> Optional[Any]:
        """Get tool executor from context services.

        Args:
            ctx: Tick context.

        Returns:
            Tool executor service or None.
        """
        if ctx.services and hasattr(ctx.services, "tool_executor"):
            return ctx.services.tool_executor
        return None

    def _validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate required parameters.

        Override in subclasses for specific validation.

        Args:
            params: Interpolated parameters.

        Returns:
            List of missing parameter names.
        """
        return []

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information."""
        info = super().debug_info()
        info["tool_name"] = self._tool_name
        info["output_key"] = self._output_key
        info["params_template"] = self._params
        return info


# =============================================================================
# Oracle Node
# =============================================================================


class Oracle(ToolLeaf):
    """Multi-source oracle query with streaming support.

    From contracts/nodes.yaml Oracle leaf:
    - First tick: initiate oracle request via OracleBridge
    - During streaming: update stream_to key with chunks
    - On completion: write full response to output key
    - Emit llm.request.complete event with token usage

    Example:
        >>> oracle = Oracle(
        ...     id="ask-question",
        ...     question="${bb.user_question}",
        ...     sources=["code", "vault"],
        ...     stream_to="partial_response",
        ...     output="oracle_answer",
        ... )
    """

    # Oracle has longer timeout by default
    DEFAULT_TIMEOUT_MS = 120000  # 2 minutes

    def __init__(
        self,
        id: str,
        question: str,
        sources: Optional[List[str]] = None,
        explain: bool = False,
        stream_to: Optional[str] = None,
        output: str = "oracle_answer",
        timeout_ms: Optional[int] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize an Oracle node.

        Args:
            id: Unique node identifier.
            question: Natural language question (supports ${bb.key}).
            sources: Filter sources (code, vault, threads).
            explain: Include reasoning in response.
            stream_to: Blackboard key for streaming chunks.
            output: Blackboard key for final response.
            timeout_ms: Override default timeout.
            name: Human-readable name (defaults to id).
            metadata: Optional metadata for debugging.
        """
        super().__init__(
            id=id,
            timeout_ms=timeout_ms or self.DEFAULT_TIMEOUT_MS,
            name=name,
            metadata=metadata,
        )

        self._question_template = question
        self._sources = sources or ["code", "vault", "threads"]
        self._explain = explain
        self._stream_to = stream_to
        self._output_key = output

        # State tracking (from nodes.yaml)
        self._chunks_received: int = 0
        self._accumulated_response: str = ""

    @classmethod
    def contract(cls) -> NodeContract:
        """Oracle contract.

        Note: Returns empty contract since actual keys are instance-specific.
        Question is provided in constructor, output key is configurable.
        """
        return NodeContract(
            description="Oracle node - multi-source query with streaming"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute the oracle query.

        Args:
            ctx: Tick context.

        Returns:
            SUCCESS when complete, RUNNING while streaming, FAILURE on error.
        """
        # First tick: initiate request
        if self._tool_start_time is None:
            return self._start_oracle_request(ctx)

        # Subsequent ticks: check completion
        return self._check_oracle_completion(ctx)

    def _start_oracle_request(self, ctx: "TickContext") -> RunStatus:
        """Start oracle request (first tick).

        Args:
            ctx: Tick context.

        Returns:
            RUNNING to continue processing, FAILURE on error.
        """
        self._tool_start_time = datetime.now(timezone.utc)
        self._chunks_received = 0
        self._accumulated_response = ""

        # Interpolate question
        if ctx.blackboard:
            question = _interpolate_string(self._question_template, ctx.blackboard)
        else:
            question = self._question_template

        if not question:
            logger.error(f"Oracle node '{self._id}' has empty question")
            self._write_error(ctx, "Question is empty")
            return RunStatus.FAILURE

        # Get oracle bridge from services
        oracle_bridge = self._get_oracle_bridge(ctx)
        if oracle_bridge is None:
            logger.error(f"No oracle bridge available for node '{self._id}'")
            self._write_error(ctx, "Oracle bridge not available")
            return RunStatus.FAILURE

        # Generate request ID
        self._request_id = f"oracle-{self._id}-{int(time.time() * 1000)}"
        ctx.add_async(self._request_id)

        # Start async request (non-streaming for now)
        # TODO: Implement streaming with stream_to updates
        try:
            # Store request parameters for async checking
            if ctx.blackboard:
                ctx.blackboard.set_internal(f"_oracle_request_{self._request_id}", {
                    "question": question,
                    "sources": self._sources,
                    "explain": self._explain,
                })

            logger.debug(
                f"Oracle node '{self._id}' started request: {self._request_id}"
            )
            return RunStatus.RUNNING

        except Exception as e:
            logger.error(f"Oracle request failed: {e}")
            self._write_error(ctx, str(e))
            return RunStatus.FAILURE

    def _check_oracle_completion(self, ctx: "TickContext") -> RunStatus:
        """Check oracle request completion.

        Args:
            ctx: Tick context.

        Returns:
            SUCCESS on completion, RUNNING while waiting, FAILURE on error.
        """
        # Check timeout
        if self._check_timeout():
            logger.error(
                f"Oracle request '{self._request_id}' timed out after {self._timeout_ms}ms"
            )
            self._write_error(ctx, f"Oracle timeout after {self._timeout_ms}ms")
            if self._request_id:
                ctx.complete_async(self._request_id)
            return RunStatus.FAILURE

        # Get oracle bridge
        oracle_bridge = self._get_oracle_bridge(ctx)
        if oracle_bridge is None:
            return RunStatus.FAILURE

        # For now, we simulate sync execution since the actual bridge is async
        # In production, this would check a result queue or use asyncio coordination
        try:
            # Retrieve stored request params
            request_params = None
            if ctx.blackboard:
                request_params = ctx.blackboard._lookup(
                    f"_oracle_request_{self._request_id}"
                )

            if request_params is None:
                # Request not found - still waiting
                return RunStatus.RUNNING

            # Execute synchronously (blocking - for MVP)
            # In production, use asyncio.run() or event loop coordination
            import asyncio

            async def run_oracle():
                return await oracle_bridge.ask_oracle(
                    question=request_params["question"],
                    sources=request_params["sources"],
                    explain=request_params["explain"],
                )

            # Try to get existing loop or create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't run sync in running loop - return RUNNING
                    # This will be resolved with proper async integration
                    return RunStatus.RUNNING
                result = loop.run_until_complete(run_oracle())
            except RuntimeError:
                # No event loop - create one
                result = asyncio.run(run_oracle())

            # Complete async operation
            ctx.complete_async(self._request_id)

            # Handle result
            return self._handle_oracle_result(ctx, result)

        except Exception as e:
            logger.error(f"Oracle query failed: {e}")
            self._write_error(ctx, str(e))
            if self._request_id:
                ctx.complete_async(self._request_id)
            return RunStatus.FAILURE

    def _handle_oracle_result(self, ctx: "TickContext", result: Dict[str, Any]) -> RunStatus:
        """Handle oracle result.

        Args:
            ctx: Tick context.
            result: Oracle response.

        Returns:
            SUCCESS or FAILURE.
        """
        duration_ms = self._get_elapsed_ms()

        # Check for error
        if result.get("error"):
            error_msg = result.get("message", "Unknown oracle error")
            self._write_error(ctx, error_msg)
            return RunStatus.FAILURE

        # Write results to blackboard
        if ctx.blackboard:
            # Write full response
            ctx.blackboard.set_internal(f"_{self._output_key}", result)

            # Write token usage
            tokens_used = result.get("tokens_used", 0)
            ctx.blackboard.set_internal("_oracle_tokens_used", tokens_used)

            # Clean up request params
            try:
                ctx.blackboard.delete(f"_oracle_request_{self._request_id}")
            except Exception:
                pass

        # TODO: Emit llm.request.complete event

        logger.debug(
            f"Oracle node '{self._id}' completed in {duration_ms:.2f}ms, "
            f"tokens: {result.get('tokens_used', 'unknown')}"
        )
        return RunStatus.SUCCESS

    def _write_error(self, ctx: "TickContext", error: str) -> None:
        """Write error to blackboard."""
        if ctx.blackboard:
            ctx.blackboard.set_internal("_oracle_error", error)

    def _get_oracle_bridge(self, ctx: "TickContext") -> Optional[Any]:
        """Get oracle bridge from services.

        Args:
            ctx: Tick context.

        Returns:
            OracleBridge instance or None.
        """
        if ctx.services and hasattr(ctx.services, "oracle_bridge"):
            return ctx.services.oracle_bridge
        return None

    def reset(self) -> None:
        """Reset oracle state."""
        super().reset()
        self._chunks_received = 0
        self._accumulated_response = ""

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information."""
        info = super().debug_info()
        info["question_template"] = self._question_template
        info["sources"] = self._sources
        info["explain"] = self._explain
        info["stream_to"] = self._stream_to
        info["output_key"] = self._output_key
        info["chunks_received"] = self._chunks_received
        return info


# =============================================================================
# CodeSearch Node
# =============================================================================


class CodeSearch(ToolLeaf):
    """Code search operations via CodeRAG.

    From contracts/nodes.yaml CodeSearch leaf:
    - Supports: search (semantic), definition, references, repo_map
    - Execute search operation via CodeRAGService
    - Results written to output blackboard key
    - Async operation, returns RUNNING until complete

    Example:
        >>> code_search = CodeSearch(
        ...     id="find-code",
        ...     operation="search",
        ...     query="${bb.search_query}",
        ...     limit=20,
        ...     output="code_results",
        ... )
    """

    # Code search timeout
    DEFAULT_TIMEOUT_MS = 60000  # 1 minute

    # Valid operations
    OPERATIONS = {"search", "definition", "references", "repo_map"}

    def __init__(
        self,
        id: str,
        operation: str,
        query: Optional[str] = None,
        limit: int = 10,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None,
        scope: Optional[str] = None,
        output: str = "code_results",
        timeout_ms: Optional[int] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a CodeSearch node.

        Args:
            id: Unique node identifier.
            operation: Operation type (search, definition, references, repo_map).
            query: Search query or symbol name (supports ${bb.key}).
            limit: Maximum results to return.
            language: Filter by language (python, typescript, etc.).
            file_pattern: Glob pattern for files.
            scope: Directory scope for repo_map.
            output: Blackboard key for results.
            timeout_ms: Override default timeout.
            name: Human-readable name (defaults to id).
            metadata: Optional metadata for debugging.

        Raises:
            ValueError: If operation is not valid.
        """
        super().__init__(
            id=id,
            timeout_ms=timeout_ms or self.DEFAULT_TIMEOUT_MS,
            name=name,
            metadata=metadata,
        )

        if operation not in self.OPERATIONS:
            raise ValueError(
                f"Invalid operation '{operation}'. "
                f"Must be one of: {self.OPERATIONS}"
            )

        self._operation = operation
        self._query_template = query
        self._limit = limit
        self._language = language
        self._file_pattern = file_pattern
        self._scope = scope
        self._output_key = output

    @classmethod
    def contract(cls) -> NodeContract:
        """CodeSearch contract.

        Note: Returns empty contract since actual keys are instance-specific.
        Query and output key are provided in constructor.
        """
        return NodeContract(
            description="CodeSearch node - code search via CodeRAG"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute code search.

        Args:
            ctx: Tick context.

        Returns:
            SUCCESS on completion, RUNNING while executing, FAILURE on error.
        """
        # First tick: start search
        if self._tool_start_time is None:
            return self._start_search(ctx)

        # Subsequent ticks: check completion
        return self._check_search_completion(ctx)

    def _start_search(self, ctx: "TickContext") -> RunStatus:
        """Start code search (first tick).

        Args:
            ctx: Tick context.

        Returns:
            RUNNING to continue, FAILURE on error.
        """
        self._tool_start_time = datetime.now(timezone.utc)

        # Interpolate query
        query = None
        if self._query_template and ctx.blackboard:
            query = _interpolate_string(self._query_template, ctx.blackboard)
        elif self._query_template:
            query = self._query_template

        # Validate query (required for most operations)
        if self._operation != "repo_map" and not query:
            logger.error(
                f"CodeSearch node '{self._id}' requires query for operation '{self._operation}'"
            )
            self._write_error(ctx, f"Query required for {self._operation}")
            return RunStatus.FAILURE

        # Get oracle bridge (which has code search methods)
        bridge = self._get_oracle_bridge(ctx)
        if bridge is None:
            logger.error(f"No code search service available for node '{self._id}'")
            self._write_error(ctx, "Code search service not available")
            return RunStatus.FAILURE

        # Generate request ID
        self._request_id = f"codesearch-{self._id}-{int(time.time() * 1000)}"
        ctx.add_async(self._request_id)

        # Store search params
        if ctx.blackboard:
            ctx.blackboard.set_internal(f"_codesearch_{self._request_id}", {
                "operation": self._operation,
                "query": query,
                "limit": self._limit,
                "language": self._language,
                "file_pattern": self._file_pattern,
                "scope": self._scope,
            })

        logger.debug(
            f"CodeSearch node '{self._id}' started: {self._operation}"
        )
        return RunStatus.RUNNING

    def _check_search_completion(self, ctx: "TickContext") -> RunStatus:
        """Check code search completion.

        Args:
            ctx: Tick context.

        Returns:
            SUCCESS on completion, RUNNING while waiting, FAILURE on error.
        """
        # Check timeout
        if self._check_timeout():
            logger.error(
                f"CodeSearch '{self._request_id}' timed out after {self._timeout_ms}ms"
            )
            self._write_error(ctx, f"Code search timeout after {self._timeout_ms}ms")
            if self._request_id:
                ctx.complete_async(self._request_id)
            return RunStatus.FAILURE

        # Get bridge
        bridge = self._get_oracle_bridge(ctx)
        if bridge is None:
            return RunStatus.FAILURE

        # Get stored params
        params = None
        if ctx.blackboard:
            params = ctx.blackboard._lookup(f"_codesearch_{self._request_id}")

        if params is None:
            return RunStatus.RUNNING

        try:
            # Execute search (sync wrapper for async)
            import asyncio

            async def run_search():
                op = params["operation"]
                if op == "search":
                    return await bridge.search_code(
                        query=params["query"],
                        limit=params["limit"],
                        language=params["language"],
                        file_pattern=params["file_pattern"],
                    )
                elif op == "definition":
                    return await bridge.find_definition(
                        symbol=params["query"],
                        scope=params["scope"],
                    )
                elif op == "references":
                    return await bridge.find_references(
                        symbol=params["query"],
                        limit=params["limit"],
                    )
                elif op == "repo_map":
                    return await bridge.get_repo_map(
                        scope=params["scope"],
                    )
                else:
                    return {"error": True, "message": f"Unknown operation: {op}"}

            # Try to execute
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return RunStatus.RUNNING
                result = loop.run_until_complete(run_search())
            except RuntimeError:
                result = asyncio.run(run_search())

            # Complete
            ctx.complete_async(self._request_id)

            # Handle result
            return self._handle_search_result(ctx, result)

        except Exception as e:
            logger.error(f"Code search failed: {e}")
            self._write_error(ctx, str(e))
            if self._request_id:
                ctx.complete_async(self._request_id)
            return RunStatus.FAILURE

    def _handle_search_result(self, ctx: "TickContext", result: Dict[str, Any]) -> RunStatus:
        """Handle code search result.

        Args:
            ctx: Tick context.
            result: Search result.

        Returns:
            SUCCESS or FAILURE.
        """
        duration_ms = self._get_elapsed_ms()

        # Check for error
        if result.get("error"):
            error_msg = result.get("message", "Unknown code search error")
            self._write_error(ctx, error_msg)
            return RunStatus.FAILURE

        # Write results
        if ctx.blackboard:
            ctx.blackboard.set_internal(f"_{self._output_key}", result)
            ctx.blackboard.set_internal("_code_search_duration_ms", duration_ms)

            # Clean up
            try:
                ctx.blackboard.delete(f"_codesearch_{self._request_id}")
            except Exception:
                pass

        logger.debug(
            f"CodeSearch node '{self._id}' completed in {duration_ms:.2f}ms"
        )
        return RunStatus.SUCCESS

    def _write_error(self, ctx: "TickContext", error: str) -> None:
        """Write error to blackboard."""
        if ctx.blackboard:
            ctx.blackboard.set_internal("_code_search_error", error)
            ctx.blackboard.set_internal("_code_search_duration_ms", self._get_elapsed_ms())

    def _get_oracle_bridge(self, ctx: "TickContext") -> Optional[Any]:
        """Get oracle bridge (has code search methods)."""
        if ctx.services and hasattr(ctx.services, "oracle_bridge"):
            return ctx.services.oracle_bridge
        return None

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information."""
        info = super().debug_info()
        info["operation"] = self._operation
        info["query_template"] = self._query_template
        info["limit"] = self._limit
        info["language"] = self._language
        info["file_pattern"] = self._file_pattern
        info["scope"] = self._scope
        info["output_key"] = self._output_key
        return info


# =============================================================================
# VaultSearch Node
# =============================================================================


class VaultSearch(ToolLeaf):
    """Search vault notes via BM25.

    From contracts/nodes.yaml VaultSearch leaf:
    - Execute search_notes MCP tool
    - Filter by tags if provided
    - Results include BM25 score and snippets

    Example:
        >>> vault_search = VaultSearch(
        ...     id="search-docs",
        ...     query="${bb.query}",
        ...     tags=["project", "design"],
        ...     limit=5,
        ...     output="notes",
        ... )
    """

    DEFAULT_TIMEOUT_MS = 30000  # 30 seconds

    def __init__(
        self,
        id: str,
        query: str,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        output: str = "notes",
        timeout_ms: Optional[int] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a VaultSearch node.

        Args:
            id: Unique node identifier.
            query: BM25 search query (supports ${bb.key}).
            tags: Filter by tags (AND logic).
            limit: Maximum results to return.
            output: Blackboard key for results.
            timeout_ms: Override default timeout.
            name: Human-readable name (defaults to id).
            metadata: Optional metadata for debugging.
        """
        super().__init__(
            id=id,
            timeout_ms=timeout_ms or self.DEFAULT_TIMEOUT_MS,
            name=name,
            metadata=metadata,
        )

        self._query_template = query
        self._tags = tags
        self._limit = limit
        self._output_key = output

    @classmethod
    def contract(cls) -> NodeContract:
        """VaultSearch contract.

        Note: Returns empty contract since actual keys are instance-specific.
        Query and output key are provided in constructor.
        """
        return NodeContract(
            description="VaultSearch node - BM25 search over vault notes"
        )

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute vault search.

        Args:
            ctx: Tick context.

        Returns:
            SUCCESS on completion, FAILURE on error.
        """
        if self._tool_start_time is None:
            self._tool_start_time = datetime.now(timezone.utc)

        # Interpolate query
        if ctx.blackboard:
            query = _interpolate_string(self._query_template, ctx.blackboard)
        else:
            query = self._query_template

        if not query:
            logger.error(f"VaultSearch node '{self._id}' has empty query")
            self._write_error(ctx, "Query is empty")
            return RunStatus.FAILURE

        # Get indexer service from services
        indexer = self._get_indexer_service(ctx)
        if indexer is None:
            logger.error(f"No indexer service available for node '{self._id}'")
            self._write_error(ctx, "Indexer service not available")
            return RunStatus.FAILURE

        try:
            # Execute search (sync)
            # Note: Need user_id from context
            user_id = self._get_user_id(ctx)

            results = indexer.search_notes(
                user_id=user_id,
                query=query,
                tags=self._tags,
                limit=self._limit,
            )

            duration_ms = self._get_elapsed_ms()

            # Write results
            if ctx.blackboard:
                ctx.blackboard.set_internal(f"_{self._output_key}", results)
                ctx.blackboard.set_internal("_vault_search_duration_ms", duration_ms)

            logger.debug(
                f"VaultSearch node '{self._id}' found {len(results)} results "
                f"in {duration_ms:.2f}ms"
            )
            return RunStatus.SUCCESS

        except Exception as e:
            logger.error(f"Vault search failed: {e}")
            self._write_error(ctx, str(e))
            return RunStatus.FAILURE

    def _write_error(self, ctx: "TickContext", error: str) -> None:
        """Write error to blackboard."""
        if ctx.blackboard:
            ctx.blackboard.set_internal("_vault_search_error", error)
            ctx.blackboard.set_internal("_vault_search_duration_ms", self._get_elapsed_ms())

    def _get_indexer_service(self, ctx: "TickContext") -> Optional[Any]:
        """Get indexer service from context."""
        if ctx.services and hasattr(ctx.services, "indexer"):
            return ctx.services.indexer
        return None

    def _get_user_id(self, ctx: "TickContext") -> str:
        """Get user ID from context.

        Falls back to 'local-dev' if not available.
        """
        if ctx.blackboard:
            # Try to get from blackboard
            identity = ctx.blackboard._lookup("identity")
            if identity and hasattr(identity, "user_id"):
                return identity.user_id
            if isinstance(identity, dict) and "user_id" in identity:
                return identity["user_id"]

        return "local-dev"

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information."""
        info = super().debug_info()
        info["query_template"] = self._query_template
        info["tags"] = self._tags
        info["limit"] = self._limit
        info["output_key"] = self._output_key
        return info


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Base class
    "ToolLeaf",
    # Tool nodes
    "Tool",
    "Oracle",
    "CodeSearch",
    "VaultSearch",
    # Helper functions
    "interpolate_params",
    # Error types
    "ToolNotFoundError",
    "ToolTimeoutError",
    "MissingToolParameterError",
    # Result type
    "ToolResult",
]
