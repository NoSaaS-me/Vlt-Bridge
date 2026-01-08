"""
Tree Definition Data Classes - Intermediate representation from Lua parsing.

These dataclasses represent the parsed tree structure before it's built
into an executable BehaviorTree. They're used by:
- TreeLoader: Parses Lua DSL into TreeDefinition
- TreeValidator: Validates TreeDefinition before building
- TreeBuilder: Builds BehaviorTree from validated TreeDefinition

From tree-loader.yaml dataclasses section.

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.7.1-2.7.5 (TreeValidator), 2.7.6-2.7.10 (TreeBuilder)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Node Type Constants
# =============================================================================


class NodeTypeEnum(str, Enum):
    """Valid node types in tree definitions."""

    # Composites
    SEQUENCE = "sequence"
    SELECTOR = "selector"
    PARALLEL = "parallel"
    FOR_EACH = "for_each"

    # Decorators
    TIMEOUT = "timeout"
    RETRY = "retry"
    GUARD = "guard"
    COOLDOWN = "cooldown"
    INVERTER = "inverter"
    ALWAYS_SUCCEED = "always_succeed"
    ALWAYS_FAIL = "always_fail"

    # Leaves
    ACTION = "action"
    CONDITION = "condition"
    SUBTREE_REF = "subtree_ref"
    SCRIPT = "script"

    # MCP leaves (may be stubs)
    TOOL = "tool"
    ORACLE = "oracle"
    CODE_SEARCH = "code_search"
    VAULT_SEARCH = "vault_search"
    LLM_CALL = "llm_call"

    @classmethod
    def is_composite(cls, node_type: str) -> bool:
        """Check if node type is a composite (1+ children)."""
        return node_type in {
            cls.SEQUENCE.value,
            cls.SELECTOR.value,
            cls.PARALLEL.value,
            cls.FOR_EACH.value,
        }

    @classmethod
    def is_decorator(cls, node_type: str) -> bool:
        """Check if node type is a decorator (exactly 1 child)."""
        return node_type in {
            cls.TIMEOUT.value,
            cls.RETRY.value,
            cls.GUARD.value,
            cls.COOLDOWN.value,
            cls.INVERTER.value,
            cls.ALWAYS_SUCCEED.value,
            cls.ALWAYS_FAIL.value,
        }

    @classmethod
    def is_leaf(cls, node_type: str) -> bool:
        """Check if node type is a leaf (0 children)."""
        return node_type in {
            cls.ACTION.value,
            cls.CONDITION.value,
            cls.SUBTREE_REF.value,
            cls.SCRIPT.value,
            cls.TOOL.value,
            cls.ORACLE.value,
            cls.CODE_SEARCH.value,
            cls.VAULT_SEARCH.value,
            cls.LLM_CALL.value,
        }

    @classmethod
    def get_expected_children(cls, node_type: str) -> str:
        """Get expected children count description for error messages."""
        if cls.is_composite(node_type):
            return "1+"
        elif cls.is_decorator(node_type):
            return "1"
        elif cls.is_leaf(node_type):
            return "0"
        else:
            return "unknown"


# =============================================================================
# Node ID Validation
# =============================================================================


# From tree-loader.yaml file_validation
NODE_ID_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")

# From tree-loader.yaml tree_name_pattern
TREE_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def is_valid_node_id(node_id: str) -> bool:
    """Check if node ID matches the required pattern.

    From tree-loader.yaml file_validation:
    - Start with letter
    - Contain only letters, numbers, underscore, hyphen
    """
    return bool(node_id and NODE_ID_PATTERN.match(node_id))


def is_valid_tree_name(tree_name: str) -> bool:
    """Check if tree name is valid.

    Same pattern as node ID validation.
    """
    return bool(tree_name and TREE_NAME_PATTERN.match(tree_name))


# =============================================================================
# NodeDefinition
# =============================================================================


@dataclass
class NodeDefinition:
    """Intermediate node representation from Lua parsing.

    From tree-loader.yaml dataclasses.NodeDefinition:
    - type: Node type string (sequence, selector, action, etc.)
    - id: Optional node ID (auto-generated if not provided)
    - name: Optional human-readable name
    - config: Node configuration dictionary
    - children: List of child NodeDefinitions
    - source_line: Line number in Lua file for error reporting

    Example:
        >>> node = NodeDefinition(
        ...     type="sequence",
        ...     id="main-sequence",
        ...     children=[
        ...         NodeDefinition(type="action", id="step-1", config={"fn": "my.action"}),
        ...         NodeDefinition(type="action", id="step-2", config={"fn": "my.other"}),
        ...     ],
        ...     source_line=10,
        ... )
    """

    type: str
    id: Optional[str] = None
    name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    children: List["NodeDefinition"] = field(default_factory=list)
    source_line: int = 0

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if self.id is None:
            # Auto-generate ID from type and source line
            self.id = f"{self.type}_{self.source_line}"

    @property
    def is_composite(self) -> bool:
        """Check if this is a composite node type."""
        return NodeTypeEnum.is_composite(self.type)

    @property
    def is_decorator(self) -> bool:
        """Check if this is a decorator node type."""
        return NodeTypeEnum.is_decorator(self.type)

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node type."""
        return NodeTypeEnum.is_leaf(self.type)

    @property
    def expected_children(self) -> str:
        """Get expected children count description."""
        return NodeTypeEnum.get_expected_children(self.type)

    def get_fn_path(self) -> Optional[str]:
        """Get function path from config for action/condition nodes."""
        return self.config.get("fn")

    def get_subtree_name(self) -> Optional[str]:
        """Get subtree name from config for subtree_ref nodes."""
        return self.config.get("tree") or self.config.get("name")

    def is_lazy_subtree(self) -> bool:
        """Check if subtree ref is lazy (resolved at runtime)."""
        return self.config.get("lazy", False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "config": self.config,
            "children": [c.to_dict() for c in self.children],
            "source_line": self.source_line,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeDefinition":
        """Create from dictionary."""
        return cls(
            type=data["type"],
            id=data.get("id"),
            name=data.get("name"),
            config=data.get("config", {}),
            children=[cls.from_dict(c) for c in data.get("children", [])],
            source_line=data.get("source_line", 0),
        )


# =============================================================================
# TreeDefinition
# =============================================================================


@dataclass
class TreeDefinition:
    """Intermediate representation from Lua parsing.

    From tree-loader.yaml dataclasses.TreeDefinition:
    - name: Tree name (used as ID)
    - description: Optional description
    - root: Root NodeDefinition
    - blackboard_schema: Key -> Pydantic model path mapping
    - config: Tree-level configuration
    - source_path: Path to source .lua file
    - source_hash: SHA256 hash of source file

    Example:
        >>> tree_def = TreeDefinition(
        ...     name="oracle-agent",
        ...     root=NodeDefinition(type="sequence", id="root", children=[...]),
        ...     source_path="/path/to/oracle-agent.lua",
        ... )
    """

    name: str
    root: NodeDefinition
    source_path: str = ""
    description: str = ""
    blackboard_schema: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    source_hash: str = ""

    def __post_init__(self) -> None:
        """Validate tree definition after creation."""
        if not self.name:
            raise ValueError("Tree name cannot be empty")
        if not is_valid_tree_name(self.name):
            raise ValueError(
                f"Invalid tree name '{self.name}': must match pattern "
                f"^[a-zA-Z][a-zA-Z0-9_-]*$"
            )
        if self.root is None:
            raise ValueError("Tree root cannot be None")

    @property
    def id(self) -> str:
        """Tree ID is derived from name."""
        return self.name

    def compute_source_hash(self) -> str:
        """Compute SHA256 hash of source file.

        Used for change detection in hot reload.
        """
        if not self.source_path:
            return ""

        try:
            path = Path(self.source_path)
            if path.exists():
                content = path.read_bytes()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            pass

        return ""

    def all_node_ids(self) -> List[str]:
        """Get all node IDs in the tree.

        Used for duplicate detection.
        """
        ids: List[str] = []
        self._collect_node_ids(self.root, ids)
        return ids

    def _collect_node_ids(self, node: NodeDefinition, ids: List[str]) -> None:
        """Recursively collect node IDs."""
        if node.id:
            ids.append(node.id)
        for child in node.children:
            self._collect_node_ids(child, ids)

    def all_subtree_refs(self) -> List[str]:
        """Get all subtree reference names.

        Used for dependency resolution.
        """
        refs: List[str] = []
        self._collect_subtree_refs(self.root, refs)
        return refs

    def _collect_subtree_refs(
        self,
        node: NodeDefinition,
        refs: List[str],
    ) -> None:
        """Recursively collect subtree references."""
        if node.type == "subtree_ref":
            subtree_name = node.get_subtree_name()
            if subtree_name and not node.is_lazy_subtree():
                refs.append(subtree_name)

        for child in node.children:
            self._collect_subtree_refs(child, refs)

    def all_fn_paths(self) -> List[str]:
        """Get all function paths referenced in the tree.

        Used for function resolution validation.
        """
        paths: List[str] = []
        self._collect_fn_paths(self.root, paths)
        return paths

    def _collect_fn_paths(
        self,
        node: NodeDefinition,
        paths: List[str],
    ) -> None:
        """Recursively collect function paths."""
        if node.type in ("action", "condition"):
            fn_path = node.get_fn_path()
            if fn_path:
                paths.append(fn_path)

        for child in node.children:
            self._collect_fn_paths(child, paths)

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            root=NodeDefinition.from_dict(data["root"]),
            blackboard_schema=data.get("blackboard_schema", {}),
            config=data.get("config", {}),
            source_path=data.get("source_path", ""),
            source_hash=data.get("source_hash", ""),
        )


# =============================================================================
# ValidationError
# =============================================================================


@dataclass
class ValidationError:
    """Error from tree validation.

    From tree-loader.yaml dataclasses.ValidationError:
    - code: Error code (E4xxx)
    - location: tree_id:node_id:line_number
    - message: Human-readable error message

    Example:
        >>> error = ValidationError(
        ...     code="E4004",
        ...     location="oracle-agent:run-research:42",
        ...     message="Subtree 'reserch-runner' not found. Did you mean 'research-runner'?",
        ... )
    """

    code: str
    location: str
    message: str

    # Optional suggestion for typos
    suggestion: Optional[str] = None

    # Optional list of available options
    available: Optional[List[str]] = None

    def __str__(self) -> str:
        """Format error for display."""
        parts = [f"[{self.code}] {self.message}"]
        parts.append(f"  Location: {self.location}")

        if self.suggestion:
            parts.append(f"  Did you mean: '{self.suggestion}'?")

        if self.available:
            parts.append("  Available:")
            for item in self.available[:5]:
                parts.append(f"    - {item}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "code": self.code,
            "location": self.location,
            "message": self.message,
        }
        if self.suggestion:
            result["suggestion"] = self.suggestion
        if self.available:
            result["available"] = self.available
        return result

    @staticmethod
    def make_location(
        tree_id: str,
        node_id: Optional[str] = None,
        line_number: Optional[int] = None,
    ) -> str:
        """Create location string from components."""
        parts = [tree_id]
        if node_id:
            parts.append(node_id)
        if line_number is not None:
            parts.append(str(line_number))
        return ":".join(parts)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "NodeTypeEnum",
    "NODE_ID_PATTERN",
    "TREE_NAME_PATTERN",
    "is_valid_node_id",
    "is_valid_tree_name",
    "NodeDefinition",
    "TreeDefinition",
    "ValidationError",
]
