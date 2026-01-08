"""
BT State Bridges - Conversion between different state representations.

This module provides bridges for converting between:
- RuleContext (plugin system) <-> OracleState (BT runtime)
- TypedBlackboard <-> Lua tables

Part of the BT Universal Runtime (spec 019).
Implements tasks 0.5.1-0.5.5 from tasks.md.

Reference:
- contracts/blackboard.yaml - Type coercion rules
- state-architecture.md - State type hierarchy
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel

from .blackboard import TypedBlackboard, lua_to_python, python_to_lua
from .composite import OracleState
from .types import (
    IdentityState,
    ConversationState,
    BudgetState,
    ToolState,
    ToolCallState,
    ToolCallStatus,
    MessageState,
)

if TYPE_CHECKING:
    from ...services.plugins.context import (
        RuleContext,
        TurnState,
        HistoryState,
        UserState,
        ProjectState,
        PluginState,
        ToolCallRecord,
    )


class RuleContextBridge:
    """Bridge for converting between RuleContext and OracleState.

    RuleContext is used by the plugin system for rule evaluation.
    OracleState is used by the BT runtime for agent execution.

    This bridge enables:
    1. Migrating from plugins to BT without breaking existing rules
    2. Sharing state between old and new systems during transition
    3. Gradual migration path

    Example:
        >>> from services.plugins.context import RuleContext, TurnState, ...
        >>> rc = RuleContext.create_minimal("user-123", "project-456")
        >>> oracle_state = RuleContextBridge.from_rule_context(rc)
        >>> oracle_state.user_id
        'user-123'

        >>> oracle_state = OracleState(user_id="user-123")
        >>> rc_dict = RuleContextBridge.to_rule_context(oracle_state)
        >>> rc_dict["user"]["id"]
        'user-123'
    """

    @staticmethod
    def from_rule_context(rc: "RuleContext") -> OracleState:
        """Convert RuleContext to OracleState.

        Extracts state from all RuleContext components and assembles
        into the composite OracleState for BT runtime consumption.

        Args:
            rc: RuleContext from plugin system

        Returns:
            OracleState with values extracted from RuleContext

        Mapping:
            - rc.turn.number -> turn_number
            - rc.turn.token_usage -> tokens_used (via budget calculation)
            - rc.turn.context_usage -> context_tokens (via context calculation)
            - rc.turn.iteration_count -> iterations_used
            - rc.user.id -> user_id
            - rc.project.id -> project_id
            - rc.history.messages -> messages
            - rc.history.tools -> pending/running/completed_tools
            - rc.history.failures -> failure_counts
        """
        # Extract identity
        user_id = rc.user.id if rc.user else ""
        project_id = rc.project.id if rc.project else ""

        # Extract turn/budget state
        turn_number = rc.turn.number if rc.turn else 1
        iteration_count = rc.turn.iteration_count if rc.turn else 0
        token_usage = rc.turn.token_usage if rc.turn else 0.0
        context_usage = rc.turn.context_usage if rc.turn else 0.0

        # Convert usage ratios to actual values (estimate based on defaults)
        token_budget = 100000
        tokens_used = int(token_usage * token_budget)
        max_context_tokens = 128000
        context_tokens = int(context_usage * max_context_tokens)

        # Extract messages
        messages: List[MessageState] = []
        if rc.history and rc.history.messages:
            for msg in rc.history.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append(MessageState(role=role, content=content))

        # Extract tool calls
        completed_tools: List[ToolCallState] = []
        failure_counts: Dict[str, int] = {}

        if rc.history:
            # Convert ToolCallRecord objects
            for i, tool_record in enumerate(rc.history.tools or []):
                status = ToolCallStatus.SUCCESS if tool_record.success else ToolCallStatus.FAILURE
                tool_call = ToolCallState(
                    tool_id=f"tool-{i}",
                    name=tool_record.name,
                    arguments=tool_record.arguments,
                    status=status,
                    result=tool_record.result if tool_record.success else None,
                    error=None if tool_record.success else (tool_record.result or "Failed"),
                    completed_at=tool_record.timestamp,
                )
                completed_tools.append(tool_call)

            # Copy failure counts
            failure_counts = dict(rc.history.failures or {})

        return OracleState(
            # Identity
            user_id=user_id,
            project_id=project_id,
            session_id="",  # Not available in RuleContext
            # Conversation
            messages=messages,
            context_tokens=context_tokens,
            max_context_tokens=max_context_tokens,
            turn_number=turn_number,
            # Budget
            token_budget=token_budget,
            tokens_used=tokens_used,
            iteration_budget=100,  # Default
            iterations_used=iteration_count,
            timeout_ms=300000,  # Default 5 minutes
            elapsed_ms=0.0,
            # Tools
            pending_tools=[],
            running_tools=[],
            completed_tools=completed_tools,
            failure_counts=failure_counts,
        )

    @staticmethod
    def to_rule_context(state: OracleState) -> Dict[str, Any]:
        """Convert OracleState to RuleContext-compatible dict.

        Returns a dictionary that can be used to construct a RuleContext
        or passed to rule evaluation systems expecting RuleContext shape.

        Args:
            state: OracleState from BT runtime

        Returns:
            Dictionary with RuleContext-compatible structure:
            {
                "turn": {"number": ..., "token_usage": ..., ...},
                "history": {"messages": [...], "tools": [...], ...},
                "user": {"id": ..., "settings": {...}},
                "project": {"id": ..., "settings": {...}},
                "state": {"_store": {...}},
            }
        """
        # Calculate usage ratios
        token_usage = (
            state.tokens_used / state.token_budget
            if state.token_budget > 0
            else 0.0
        )
        context_usage = (
            state.context_tokens / state.max_context_tokens
            if state.max_context_tokens > 0
            else 0.0
        )

        # Build turn state
        turn = {
            "number": max(1, state.turn_number),
            "token_usage": min(1.0, max(0.0, token_usage)),
            "context_usage": min(1.0, max(0.0, context_usage)),
            "iteration_count": state.iterations_used,
        }

        # Build messages list
        messages = []
        for msg in state.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        # Build tools list from completed_tools
        tools = []
        for tool in state.completed_tools:
            tools.append({
                "name": tool.name,
                "arguments": tool.arguments,
                "result": tool.result if tool.status == ToolCallStatus.SUCCESS else tool.error,
                "success": tool.status == ToolCallStatus.SUCCESS,
                "timestamp": tool.completed_at.isoformat() if tool.completed_at else None,
            })

        # Build history state
        history = {
            "messages": messages,
            "tools": tools,
            "failures": dict(state.failure_counts),
        }

        # Build user state
        user = {
            "id": state.user_id,
            "settings": {},  # Plugin state not stored in OracleState
        }

        # Build project state
        project = {
            "id": state.project_id,
            "settings": {},  # Project settings not stored in OracleState
        }

        # Build plugin state (empty - use separate mechanism)
        plugin_state = {
            "_store": {},
        }

        return {
            "turn": turn,
            "history": history,
            "user": user,
            "project": project,
            "state": plugin_state,
        }


class LuaStateBridge:
    """Bridge for converting between TypedBlackboard and Lua-compatible data.

    Lua uses different type conventions than Python:
    - All numbers are floats
    - Tables can be arrays (1-indexed) or dicts
    - No None distinction (just nil)

    This bridge handles type coercion per blackboard.yaml specification.

    Example:
        >>> bb = TypedBlackboard()
        >>> bb.register("count", IntModel)
        >>> bb.set("count", IntModel(value=42))
        >>> lua_data = LuaStateBridge.to_lua_table(bb)
        >>> lua_data["count"]["value"]
        42.0  # float, not int

        >>> LuaStateBridge.from_lua_result({"value": 100.0}, bb, ["count"])
        # bb now has count.value = 100
    """

    @staticmethod
    def to_lua_table(bb: TypedBlackboard) -> Dict[str, Any]:
        """Convert TypedBlackboard to Lua-compatible dict.

        Creates a snapshot of the blackboard suitable for passing to Lua.
        All values are converted per blackboard.yaml type coercion rules.

        Args:
            bb: TypedBlackboard to snapshot

        Returns:
            Dictionary suitable for Lua consumption:
            - Pydantic models -> dicts
            - All numbers -> floats
            - Lists preserved
            - Nested structures recursively converted

        Note:
            This creates a COPY, not a view. Changes in Lua do not
            automatically reflect back to the blackboard.
        """
        # Get snapshot (includes parent scopes)
        snapshot = bb.snapshot()

        # Convert all values for Lua consumption
        result = {}
        for key, value in snapshot.items():
            result[key] = python_to_lua(value)

        return result

    @staticmethod
    def from_lua_result(
        result: Any,
        bb: TypedBlackboard,
        output_keys: List[str],
    ) -> None:
        """Update blackboard from Lua return value.

        Processes a Lua script's return value and updates the blackboard
        with values for the declared output keys.

        Args:
            result: Return value from Lua script (typically a table)
            bb: TypedBlackboard to update
            output_keys: List of keys that may be updated from result

        Behavior:
            - If result is a dict, extracts values for each output_key
            - Values are converted from Lua types to Python
            - Only registered and declared output keys are written
            - Unregistered keys are ignored with warning

        Example:
            Lua script returns: {count = 42.0, name = "test"}
            output_keys = ["count", "name"]
            -> bb.set("count", ...) and bb.set("name", ...) are called

        Type Coercion (from Lua):
            - number -> float (always!)
            - boolean -> bool
            - string -> str
            - table (array) -> list
            - table (dict) -> dict
            - nil -> None
        """
        if result is None:
            return

        # Convert Lua result to Python
        py_result = lua_to_python(result)

        # If result is not a dict, nothing to extract
        if not isinstance(py_result, dict):
            return

        # Extract values for declared output keys
        for key in output_keys:
            if key not in py_result:
                continue

            value = py_result[key]

            # Skip if key not registered (we can't set it anyway)
            if not bb._get_registered_schema(key):
                import logging
                logging.getLogger(__name__).warning(
                    f"Lua result contains key '{key}' but it's not registered in blackboard"
                )
                continue

            # Get the schema and try to construct/validate
            schema = bb._get_registered_schema(key)
            if schema:
                try:
                    # Try to validate/construct the value
                    if isinstance(value, dict):
                        validated_value = schema.model_validate(value)
                    elif isinstance(value, BaseModel):
                        validated_value = value
                    else:
                        # Try to construct from scalar or coerce
                        validated_value = schema.model_validate(value)

                    # Use the blackboard's set method for proper tracking
                    result_set = bb.set(key, validated_value)
                    if result_set.is_error:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Failed to set '{key}' from Lua result: {result_set.error}"
                        )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Failed to validate '{key}' from Lua result: {e}"
                    )

    @staticmethod
    def extract_status_from_lua(result: Any) -> Optional[str]:
        """Extract status field from Lua script return.

        Lua scripts conventionally return {status = "success"/"failure"/"running", ...}
        This helper extracts and normalizes the status field.

        Args:
            result: Lua return value (typically a table)

        Returns:
            Status string ("success", "failure", "running") or None if not present.
            Returns lowercase normalized version.
        """
        if result is None:
            return None

        py_result = lua_to_python(result)
        if not isinstance(py_result, dict):
            return None

        status = py_result.get("status")
        if status is None:
            return None

        # Normalize to lowercase string
        return str(status).lower()

    @staticmethod
    def extract_error_from_lua(result: Any) -> Optional[str]:
        """Extract error/reason field from Lua script return.

        Lua scripts returning failure can include a reason:
        {status = "failure", reason = "Something went wrong"}

        Args:
            result: Lua return value

        Returns:
            Error reason string or None if not present
        """
        if result is None:
            return None

        py_result = lua_to_python(result)
        if not isinstance(py_result, dict):
            return None

        # Check common error field names
        for field in ["reason", "error", "message"]:
            if field in py_result:
                return str(py_result[field])

        return None


__all__ = [
    "RuleContextBridge",
    "LuaStateBridge",
]
