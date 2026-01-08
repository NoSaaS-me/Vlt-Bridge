"""
BT.* Lua API - Functions for building behavior trees in Lua DSL.

This module provides the BT.* namespace functions that Lua scripts use
to define behavior trees. The functions create NodeDefinition and TreeDefinition
objects that are later built into executable BehaviorTree instances.

From tree-loader.yaml _inject_bt_api:
- BT.tree(name, config)
- BT.sequence(children)
- BT.selector(children)
- BT.parallel(config, children)
- BT.action(name, config)
- BT.condition(name, config)
- BT.llm_call(config)
- BT.subtree_ref(name, config?)
- BT.for_each(key, config)
- BT.script(name, config)
- BT.timeout(ms, child)
- BT.retry(max, child)
- BT.guard(condition, child)
- BT.cooldown(ms, child)
- BT.inverter(child)
- BT.always_succeed(child)
- BT.always_fail(child)
- BT.contract(config)

MCP Integration (from nodes.yaml):
- BT.tool(tool_name, config)
- BT.oracle(config)
- BT.code_search(config)
- BT.vault_search(config)

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.5.1-2.5.8 from tasks.md
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ..state.blackboard import lua_to_python


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NodeDefinition:
    """Intermediate node representation from Lua parsing.

    From tree-loader.yaml NodeDefinition:
    - type: Node type (sequence, selector, action, etc.)
    - id: Unique identifier (auto-generated if not provided)
    - name: Human-readable name
    - config: Node-specific configuration
    - children: Child NodeDefinitions
    - source_line: Line number in Lua file for debugging
    """

    type: str  # sequence, selector, parallel, action, condition, etc.
    id: Optional[str] = None
    name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    children: List["NodeDefinition"] = field(default_factory=list)
    source_line: Optional[int] = None

    def __post_init__(self):
        """Auto-generate ID if not provided."""
        if self.id is None:
            self.id = f"{self.type}_{uuid.uuid4().hex[:8]}"
        if self.name is None:
            self.name = self.id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "config": self.config,
            "children": [child.to_dict() for child in self.children],
            "source_line": self.source_line,
        }


@dataclass
class TreeDefinition:
    """Intermediate tree representation from Lua parsing.

    From tree-loader.yaml TreeDefinition:
    - name: Tree name (used as ID)
    - root: Root NodeDefinition
    - description: Human-readable description
    - blackboard_schema: Key -> Pydantic model path mapping
    - config: Tree-level configuration
    - source_path: Path to source file
    - source_hash: SHA256 of source content
    """

    name: str
    root: NodeDefinition
    description: str = ""
    blackboard_schema: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""
    source_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "root": self.root.to_dict(),
            "blackboard_schema": self.blackboard_schema,
            "config": self.config,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
        }


# =============================================================================
# BTApiBuilder
# =============================================================================


class BTApiBuilder:
    """Builds the BT.* Lua API functions.

    This class provides methods that construct NodeDefinition and TreeDefinition
    objects from Lua function calls. These definitions are later built into
    executable BehaviorTree instances by TreeBuilder.

    The functions are designed to be injected into the Lua environment as:
        BT.tree(name, config)
        BT.sequence(children)
        etc.

    Example:
        >>> builder = BTApiBuilder()
        >>> api = builder.build_api()
        >>> # In Lua: local tree = BT.tree("my-tree", {root = BT.sequence({...})})
        >>> tree_def = api["tree"]("my-tree", {"root": root_node, "description": "..."})
        >>> tree_def.name
        'my-tree'
    """

    def __init__(self, source_path: str = "<string>") -> None:
        """Initialize the API builder.

        Args:
            source_path: Source file path for error reporting.
        """
        self._source_path = source_path
        self._line_tracker: Optional[Callable[[], int]] = None

    def set_line_tracker(self, tracker: Callable[[], int]) -> None:
        """Set a callback to get current Lua line number.

        Args:
            tracker: Function that returns current line number.
        """
        self._line_tracker = tracker

    def _get_current_line(self) -> Optional[int]:
        """Get current source line if tracker is set."""
        if self._line_tracker:
            try:
                return self._line_tracker()
            except Exception:
                pass
        return None

    def build_api(self) -> Dict[str, Any]:
        """Return dict of BT.* functions to inject into Lua env.

        Returns:
            Dictionary with all BT.* function implementations.
        """
        return {
            # Tree definition
            "tree": self._bt_tree,
            # Composite nodes
            "sequence": self._bt_sequence,
            "selector": self._bt_selector,
            "parallel": self._bt_parallel,
            "for_each": self._bt_for_each,
            # Leaf nodes
            "action": self._bt_action,
            "condition": self._bt_condition,
            "llm_call": self._bt_llm_call,
            "subtree_ref": self._bt_subtree_ref,
            "script": self._bt_script,
            # Decorator nodes
            "timeout": self._bt_timeout,
            "retry": self._bt_retry,
            "guard": self._bt_guard,
            "cooldown": self._bt_cooldown,
            "inverter": self._bt_inverter,
            "always_succeed": self._bt_always_succeed,
            "always_fail": self._bt_always_fail,
            # Contract declaration
            "contract": self._bt_contract,
            # MCP integration (from nodes.yaml)
            "tool": self._bt_tool,
            "oracle": self._bt_oracle,
            "code_search": self._bt_code_search,
            "vault_search": self._bt_vault_search,
        }

    # =========================================================================
    # Tree Definition
    # =========================================================================

    def _bt_tree(
        self,
        name: Any,
        config: Any = None,
    ) -> TreeDefinition:
        """Create a tree definition.

        Lua: BT.tree("oracle-agent", {
            description = "Main Oracle agent tree",
            root = BT.sequence({...}),
            blackboard = {
                query = "QueryModel",
                response = "ResponseModel",
            }
        })

        Args:
            name: Tree name (string).
            config: Tree configuration table with root, description, blackboard.

        Returns:
            TreeDefinition object.

        Raises:
            ValueError: If name or root is missing.
        """
        # Convert Lua types
        name_str = str(name) if name else ""
        if not name_str:
            raise ValueError("BT.tree() requires a name")

        config_dict = self._to_python_dict(config) or {}

        # Extract root node
        root = config_dict.get("root")
        if root is None:
            raise ValueError(f"BT.tree('{name_str}') requires a 'root' node")

        # Ensure root is a NodeDefinition
        if not isinstance(root, NodeDefinition):
            raise ValueError(
                f"BT.tree('{name_str}') root must be a node definition "
                f"(e.g., BT.sequence, BT.selector), got {type(root).__name__}"
            )

        # Extract optional fields
        description = str(config_dict.get("description", ""))

        # Extract blackboard schema mapping
        blackboard_raw = config_dict.get("blackboard", {})
        blackboard_schema: Dict[str, str] = {}
        if isinstance(blackboard_raw, dict):
            for key, schema_path in blackboard_raw.items():
                blackboard_schema[str(key)] = str(schema_path)

        # Extract other config (excluding special keys)
        tree_config = {
            k: v for k, v in config_dict.items()
            if k not in ("root", "description", "blackboard")
        }

        return TreeDefinition(
            name=name_str,
            root=root,
            description=description,
            blackboard_schema=blackboard_schema,
            config=tree_config,
            source_path=self._source_path,
            source_hash="",  # Set by TreeLoader after parsing
        )

    # =========================================================================
    # Composite Nodes
    # =========================================================================

    def _bt_sequence(self, children: Any, config: Any = None) -> NodeDefinition:
        """Create a sequence node.

        Lua: BT.sequence({
            BT.action("step1", {...}),
            BT.action("step2", {...}),
        })

        Args:
            children: List/table of child nodes.
            config: Optional node configuration.

        Returns:
            NodeDefinition for sequence.
        """
        return self._make_composite("sequence", children, config)

    def _bt_selector(self, children: Any, config: Any = None) -> NodeDefinition:
        """Create a selector node.

        Lua: BT.selector({
            BT.condition("check", {...}),
            BT.action("fallback", {...}),
        })

        Args:
            children: List/table of child nodes.
            config: Optional node configuration.

        Returns:
            NodeDefinition for selector.
        """
        return self._make_composite("selector", children, config)

    def _bt_parallel(self, config: Any, children: Any = None) -> NodeDefinition:
        """Create a parallel node.

        Lua: BT.parallel({
            policy = "require_all",
            required_successes = 2,
            merge_strategy = "last_wins",
        }, {
            BT.action("search_code", {...}),
            BT.action("search_vault", {...}),
        })

        Args:
            config: Parallel configuration (policy, merge_strategy, etc.).
            children: List/table of child nodes.

        Returns:
            NodeDefinition for parallel.
        """
        # Handle case where children is passed as first arg
        if children is None and self._is_node_list(config):
            children = config
            config = {}

        config_dict = self._to_python_dict(config) or {}
        child_list = self._extract_children(children)

        return NodeDefinition(
            type="parallel",
            id=config_dict.get("id"),
            name=config_dict.get("name"),
            config={
                k: v for k, v in config_dict.items()
                if k not in ("id", "name")
            },
            children=child_list,
            source_line=self._get_current_line(),
        )

    def _bt_for_each(self, key: Any, config: Any = None) -> NodeDefinition:
        """Create a for_each node.

        Lua: BT.for_each("results", {
            item_key = "current_result",
            children = { BT.action("process", {...}) },
            continue_on_failure = true,
        })

        Args:
            key: Collection blackboard key.
            config: Configuration including children, item_key, etc.

        Returns:
            NodeDefinition for for_each.
        """
        config_dict = self._to_python_dict(config) or {}
        child_list = self._extract_children(config_dict.get("children", []))

        return NodeDefinition(
            type="for_each",
            id=config_dict.get("id"),
            name=config_dict.get("name"),
            config={
                "collection_key": str(key),
                "item_key": config_dict.get("item_key", "item"),
                "continue_on_failure": bool(config_dict.get("continue_on_failure", False)),
                "min_items": int(config_dict.get("min_items", 0)),
            },
            children=child_list,
            source_line=self._get_current_line(),
        )

    # =========================================================================
    # Leaf Nodes
    # =========================================================================

    def _bt_action(self, name: Any, config: Any = None) -> NodeDefinition:
        """Create an action node.

        Lua: BT.action("load_context", {
            fn = "oracle.actions.load_context",
        })

        Args:
            name: Action name (also used as ID if not specified).
            config: Action configuration with fn path.

        Returns:
            NodeDefinition for action.
        """
        name_str = str(name) if name else f"action_{uuid.uuid4().hex[:8]}"
        config_dict = self._to_python_dict(config) or {}

        return NodeDefinition(
            type="action",
            id=config_dict.get("id", name_str),
            name=config_dict.get("name", name_str),
            config={
                k: v for k, v in config_dict.items()
                if k not in ("id", "name")
            },
            children=[],
            source_line=self._get_current_line(),
        )

    def _bt_condition(self, name: Any, config: Any = None) -> NodeDefinition:
        """Create a condition node.

        Lua: BT.condition("has_budget", {
            condition = "bb.budget > 0",
        })

        Args:
            name: Condition name.
            config: Configuration with condition expression.

        Returns:
            NodeDefinition for condition.
        """
        name_str = str(name) if name else f"condition_{uuid.uuid4().hex[:8]}"
        config_dict = self._to_python_dict(config) or {}

        return NodeDefinition(
            type="condition",
            id=config_dict.get("id", name_str),
            name=config_dict.get("name", name_str),
            config={
                k: v for k, v in config_dict.items()
                if k not in ("id", "name")
            },
            children=[],
            source_line=self._get_current_line(),
        )

    def _bt_llm_call(self, config: Any) -> NodeDefinition:
        """Create an LLM call node.

        Lua: BT.llm_call({
            model = "claude-3-opus",
            prompt_key = "prompt",
            response_key = "response",
            stream_to = "partial",
        })

        Args:
            config: LLM call configuration.

        Returns:
            NodeDefinition for llm_call.
        """
        config_dict = self._to_python_dict(config) or {}

        return NodeDefinition(
            type="llm_call",
            id=config_dict.get("id"),
            name=config_dict.get("name", "llm_call"),
            config={
                k: v for k, v in config_dict.items()
                if k not in ("id", "name")
            },
            children=[],
            source_line=self._get_current_line(),
        )

    def _bt_subtree_ref(self, name: Any, config: Any = None) -> NodeDefinition:
        """Create a subtree reference node.

        Lua: BT.subtree_ref("research-runner", {
            lazy = true,
        })

        Args:
            name: Name of tree to reference.
            config: Optional configuration (lazy, etc.).

        Returns:
            NodeDefinition for subtree_ref.
        """
        name_str = str(name) if name else ""
        if not name_str:
            raise ValueError("BT.subtree_ref() requires a tree name")

        config_dict = self._to_python_dict(config) or {}

        return NodeDefinition(
            type="subtree_ref",
            id=config_dict.get("id", f"ref_{name_str}"),
            name=config_dict.get("name", f"subtree:{name_str}"),
            config={
                "tree_name": name_str,
                "lazy": bool(config_dict.get("lazy", False)),
            },
            children=[],
            source_line=self._get_current_line(),
        )

    def _bt_script(self, name: Any, config: Any = None) -> NodeDefinition:
        """Create a script node.

        Lua: BT.script("compute", {
            lua = [[
                local x = bb.get("input")
                bb.set("output", x * 2)
                return {status = "success"}
            ]],
        })

        Or with file reference:
        Lua: BT.script("process", {
            file = "scripts/process.lua",
        })

        Args:
            name: Script name.
            config: Configuration with lua code or file path.

        Returns:
            NodeDefinition for script.
        """
        name_str = str(name) if name else f"script_{uuid.uuid4().hex[:8]}"
        config_dict = self._to_python_dict(config) or {}

        return NodeDefinition(
            type="script",
            id=config_dict.get("id", name_str),
            name=config_dict.get("name", name_str),
            config={
                k: v for k, v in config_dict.items()
                if k not in ("id", "name")
            },
            children=[],
            source_line=self._get_current_line(),
        )

    # =========================================================================
    # Decorator Nodes
    # =========================================================================

    def _bt_timeout(self, ms: Any, child: Any) -> NodeDefinition:
        """Create a timeout decorator.

        Lua: BT.timeout(30000, BT.action("api_call", {...}))

        Args:
            ms: Timeout in milliseconds.
            child: Child node to wrap.

        Returns:
            NodeDefinition for timeout.
        """
        return self._make_decorator("timeout", child, {"timeout_ms": int(ms or 0)})

    def _bt_retry(self, max_retries: Any, child: Any, config: Any = None) -> NodeDefinition:
        """Create a retry decorator.

        Lua: BT.retry(3, BT.action("flaky_op", {...}), {backoff_ms = 1000})

        Args:
            max_retries: Maximum retry attempts.
            child: Child node to wrap.
            config: Optional additional config (backoff_ms, etc.).

        Returns:
            NodeDefinition for retry.
        """
        config_dict = self._to_python_dict(config) or {}
        config_dict["max_retries"] = int(max_retries or 0)
        return self._make_decorator("retry", child, config_dict)

    def _bt_guard(self, condition: Any, child: Any) -> NodeDefinition:
        """Create a guard decorator.

        Lua: BT.guard("bb.budget > 0", BT.action("expensive_op", {...}))

        Args:
            condition: Guard condition (Lua expression or callable path).
            child: Child node to wrap.

        Returns:
            NodeDefinition for guard.
        """
        return self._make_decorator("guard", child, {"condition": condition})

    def _bt_cooldown(self, ms: Any, child: Any) -> NodeDefinition:
        """Create a cooldown decorator.

        Lua: BT.cooldown(5000, BT.action("rate_limited", {...}))

        Args:
            ms: Cooldown period in milliseconds.
            child: Child node to wrap.

        Returns:
            NodeDefinition for cooldown.
        """
        return self._make_decorator("cooldown", child, {"cooldown_ms": int(ms or 0)})

    def _bt_inverter(self, child: Any) -> NodeDefinition:
        """Create an inverter decorator.

        Lua: BT.inverter(BT.condition("is_empty", {...}))

        Args:
            child: Child node to wrap.

        Returns:
            NodeDefinition for inverter.
        """
        return self._make_decorator("inverter", child, {})

    def _bt_always_succeed(self, child: Any) -> NodeDefinition:
        """Create an always_succeed decorator.

        Lua: BT.always_succeed(BT.action("optional_log", {...}))

        Args:
            child: Child node to wrap.

        Returns:
            NodeDefinition for always_succeed.
        """
        return self._make_decorator("always_succeed", child, {})

    def _bt_always_fail(self, child: Any) -> NodeDefinition:
        """Create an always_fail decorator.

        Lua: BT.always_fail(BT.action("force_retry", {...}))

        Args:
            child: Child node to wrap.

        Returns:
            NodeDefinition for always_fail.
        """
        return self._make_decorator("always_fail", child, {})

    # =========================================================================
    # Contract Declaration
    # =========================================================================

    def _bt_contract(self, config: Any) -> Dict[str, Any]:
        """Declare a node contract.

        Lua: BT.contract({
            inputs = {session_id = "SessionIdModel"},
            optional_inputs = {limit = "LimitModel"},
            outputs = {context = "ContextModel"},
            description = "Load conversation context",
        })

        This is used by Script nodes to declare their contracts in Lua.

        Args:
            config: Contract configuration.

        Returns:
            Contract configuration dictionary.
        """
        config_dict = self._to_python_dict(config) or {}
        return {
            "inputs": config_dict.get("inputs", {}),
            "optional_inputs": config_dict.get("optional_inputs", {}),
            "outputs": config_dict.get("outputs", {}),
            "description": config_dict.get("description", ""),
        }

    # =========================================================================
    # MCP Integration (from nodes.yaml)
    # =========================================================================

    def _bt_tool(self, tool_name: Any, config: Any = None) -> NodeDefinition:
        """Create a tool node for MCP tool execution.

        Lua: BT.tool("search_notes", {
            query = "${bb.user_query}",
            limit = 10,
            output = "search_results",
        })

        Args:
            tool_name: MCP tool name from registry.
            config: Tool parameters (supports ${bb.key} interpolation).

        Returns:
            NodeDefinition for tool.
        """
        name_str = str(tool_name) if tool_name else ""
        if not name_str:
            raise ValueError("BT.tool() requires a tool name")

        config_dict = self._to_python_dict(config) or {}

        return NodeDefinition(
            type="tool",
            id=config_dict.get("id", f"tool_{name_str}"),
            name=config_dict.get("name", f"tool:{name_str}"),
            config={
                "tool_name": name_str,
                "params": {k: v for k, v in config_dict.items()
                          if k not in ("id", "name", "output", "timeout")},
                "output": config_dict.get("output", f"{name_str}_result"),
                "timeout": config_dict.get("timeout"),
            },
            children=[],
            source_line=self._get_current_line(),
        )

    def _bt_oracle(self, config: Any) -> NodeDefinition:
        """Create an Oracle query node.

        Lua: BT.oracle({
            question = "${bb.user_question}",
            sources = {"code", "vault"},
            stream_to = "partial_response",
            output = "oracle_answer",
        })

        Args:
            config: Oracle configuration.

        Returns:
            NodeDefinition for oracle.
        """
        config_dict = self._to_python_dict(config) or {}

        return NodeDefinition(
            type="oracle",
            id=config_dict.get("id"),
            name=config_dict.get("name", "oracle"),
            config={
                k: v for k, v in config_dict.items()
                if k not in ("id", "name")
            },
            children=[],
            source_line=self._get_current_line(),
        )

    def _bt_code_search(self, config: Any) -> NodeDefinition:
        """Create a code search node.

        Lua: BT.code_search({
            operation = "search",
            query = "${bb.search_query}",
            limit = 20,
            output = "code_results",
        })

        Args:
            config: Code search configuration.

        Returns:
            NodeDefinition for code_search.
        """
        config_dict = self._to_python_dict(config) or {}

        return NodeDefinition(
            type="code_search",
            id=config_dict.get("id"),
            name=config_dict.get("name", "code_search"),
            config={
                k: v for k, v in config_dict.items()
                if k not in ("id", "name")
            },
            children=[],
            source_line=self._get_current_line(),
        )

    def _bt_vault_search(self, config: Any) -> NodeDefinition:
        """Create a vault search node.

        Lua: BT.vault_search({
            query = "${bb.query}",
            tags = {"project", "design"},
            limit = 5,
            output = "notes",
        })

        Args:
            config: Vault search configuration.

        Returns:
            NodeDefinition for vault_search.
        """
        config_dict = self._to_python_dict(config) or {}

        return NodeDefinition(
            type="vault_search",
            id=config_dict.get("id"),
            name=config_dict.get("name", "vault_search"),
            config={
                k: v for k, v in config_dict.items()
                if k not in ("id", "name")
            },
            children=[],
            source_line=self._get_current_line(),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _to_python_dict(self, value: Any) -> Optional[Dict[str, Any]]:
        """Convert Lua table or value to Python dict.

        Args:
            value: Lua table or Python value.

        Returns:
            Python dictionary or None.
        """
        if value is None:
            return None

        if isinstance(value, dict):
            return value

        # Try lua_to_python conversion
        try:
            converted = lua_to_python(value)
            if isinstance(converted, dict):
                return converted
        except (ValueError, TypeError):
            pass

        # Try items() for table-like objects
        if hasattr(value, "items"):
            try:
                return dict(value.items())
            except Exception:
                pass

        return None

    def _extract_children(self, children: Any) -> List[NodeDefinition]:
        """Extract child NodeDefinitions from Lua table.

        Args:
            children: Lua table or list of children.

        Returns:
            List of NodeDefinition objects.
        """
        if children is None:
            return []

        if isinstance(children, list):
            return [c for c in children if isinstance(c, NodeDefinition)]

        # Try to iterate Lua table
        result: List[NodeDefinition] = []
        try:
            # Lua tables use 1-based indexing
            if hasattr(children, "values"):
                for child in children.values():
                    if isinstance(child, NodeDefinition):
                        result.append(child)
            elif hasattr(children, "__iter__"):
                for child in children:
                    if isinstance(child, NodeDefinition):
                        result.append(child)
        except Exception:
            pass

        return result

    def _is_node_list(self, value: Any) -> bool:
        """Check if value looks like a list of nodes.

        Args:
            value: Value to check.

        Returns:
            True if value appears to be a node list.
        """
        if isinstance(value, list):
            return len(value) > 0 and isinstance(value[0], NodeDefinition)

        # Check Lua table
        try:
            if hasattr(value, "values"):
                first = next(iter(value.values()), None)
                return isinstance(first, NodeDefinition)
        except Exception:
            pass

        return False

    def _make_composite(
        self,
        node_type: str,
        children: Any,
        config: Any = None,
    ) -> NodeDefinition:
        """Create a composite node definition.

        Args:
            node_type: Type of composite (sequence, selector).
            children: Child nodes.
            config: Optional configuration.

        Returns:
            NodeDefinition for the composite.
        """
        config_dict = self._to_python_dict(config) or {}
        child_list = self._extract_children(children)

        return NodeDefinition(
            type=node_type,
            id=config_dict.get("id"),
            name=config_dict.get("name"),
            config={k: v for k, v in config_dict.items() if k not in ("id", "name")},
            children=child_list,
            source_line=self._get_current_line(),
        )

    def _make_decorator(
        self,
        node_type: str,
        child: Any,
        config: Dict[str, Any],
    ) -> NodeDefinition:
        """Create a decorator node definition.

        Args:
            node_type: Type of decorator.
            child: Child node to wrap.
            config: Decorator configuration.

        Returns:
            NodeDefinition for the decorator.
        """
        if not isinstance(child, NodeDefinition):
            raise ValueError(
                f"BT.{node_type}() requires a node definition as child, "
                f"got {type(child).__name__}"
            )

        return NodeDefinition(
            type=node_type,
            id=config.get("id"),
            name=config.get("name"),
            config={k: v for k, v in config.items() if k not in ("id", "name")},
            children=[child],
            source_line=self._get_current_line(),
        )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_source_hash(content: str) -> str:
    """Compute SHA256 hash of source content.

    Args:
        content: Source file content.

    Returns:
        Hex-encoded SHA256 hash.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "NodeDefinition",
    "TreeDefinition",
    "BTApiBuilder",
    "compute_source_hash",
]
