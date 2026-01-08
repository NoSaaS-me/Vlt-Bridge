"""
Lua DSL Integration

Contains BT.* Lua API, tree loader, validator, builder, and registry.

Components:
- LuaSandbox: Secure Lua execution environment with blocked dangerous modules
- LuaExecutionResult: Result of Lua code execution
- BTApiBuilder: Builds the BT.* Lua API functions for tree definitions
- NodeDefinition: Intermediate node representation from Lua parsing
- TreeDefinition: Intermediate tree representation from Lua parsing
- TreeLoader: Loads behavior tree definitions from Lua DSL files
- TreeValidator: Validates tree definitions before building
- TreeBuilder: Builds executable BehaviorTree from TreeDefinition
- TreeRegistry: Central registry with hot reload support

Error codes:
- E2004: Invalid node ID
- E2005: Duplicate node ID
- E2006: Node type mismatch (wrong child count)
- E3001: Tree not found
- E3002: Circular reference
- E3005: Reload failed
- E4001: File not found
- E4002: Invalid tree definition
- E4003: Function not found
- E4004: Subtree not found
- E5001: Lua syntax error
- E5002: Lua runtime error
- E5003: Lua timeout
- E7001: Sandbox violation
- E7002: Path traversal

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.4.1-2.4.6, 2.5.1-2.5.8, 2.6.1-2.6.6, 2.7.1-2.7.10, 2.8.1-2.8.8 from tasks.md
"""

# Sandbox (tasks 2.4.x)
from .sandbox import LuaSandbox, LuaExecutionResult, ERROR_CODES

# BT.* API (tasks 2.5.x)
from .api import (
    NodeDefinition,
    TreeDefinition,
    BTApiBuilder,
    compute_source_hash,
)

# TreeLoader (tasks 2.6.x)
from .loader import (
    TreeLoader,
    TreeLoadError,
    validate_tree_name,
    validate_node_id,
)

# Definitions - enhanced data classes (tasks 2.7.x)
from .definitions import (
    NodeTypeEnum,
    NODE_ID_PATTERN,
    TREE_NAME_PATTERN,
    is_valid_node_id,
    is_valid_tree_name,
    ValidationError,
)

# Validator (tasks 2.7.1-2.7.5)
from .validator import TreeValidator

# Builder (tasks 2.7.6-2.7.10)
from .builder import (
    TreeBuilder,
    TreeBuildError,
    DuplicateNodeIdError,
    NODE_BUILDERS,
)

# Registry with hot reload (tasks 2.8.1-2.8.8)
from .registry import (
    ReloadPolicy,
    TreeRegistry,
    TreeLoadError as RegistryLoadError,
    TreeValidationError,
    TreeInUseError,
    SecurityError,
)

__all__ = [
    # Sandbox (2.4.x)
    "LuaSandbox",
    "LuaExecutionResult",
    "ERROR_CODES",
    # BT.* API (2.5.x)
    "NodeDefinition",
    "TreeDefinition",
    "BTApiBuilder",
    "compute_source_hash",
    # TreeLoader (2.6.x)
    "TreeLoader",
    "TreeLoadError",
    "validate_tree_name",
    "validate_node_id",
    # Definitions (2.7.x)
    "NodeTypeEnum",
    "NODE_ID_PATTERN",
    "TREE_NAME_PATTERN",
    "is_valid_node_id",
    "is_valid_tree_name",
    "ValidationError",
    # Validator (2.7.1-2.7.5)
    "TreeValidator",
    # Builder (2.7.6-2.7.10)
    "TreeBuilder",
    "TreeBuildError",
    "DuplicateNodeIdError",
    "NODE_BUILDERS",
    # Registry (2.8.1-2.8.8)
    "ReloadPolicy",
    "TreeRegistry",
    "RegistryLoadError",
    "TreeValidationError",
    "TreeInUseError",
    "SecurityError",
]
