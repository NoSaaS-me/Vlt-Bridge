"""
TreeValidator - Validates tree definitions before building.

Run at load time to catch errors early. Performs the following checks:
- All fn paths resolve to callables (E4003)
- All subtree_refs exist or are marked lazy (E4004)
- No circular subtree references (E3002)
- Required node properties present
- Blackboard schema valid
- Node IDs unique (E2005)
- Node types have correct child counts (E2006)

From tree-loader.yaml TreeValidator interface.

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.7.1-2.7.5 from tasks.md
"""

from __future__ import annotations

import importlib
import logging
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from .definitions import (
    NodeDefinition,
    TreeDefinition,
    ValidationError,
    is_valid_node_id,
    NodeTypeEnum,
)

if TYPE_CHECKING:
    from .registry import TreeRegistry

logger = logging.getLogger(__name__)


class TreeValidator:
    """Validates tree definitions before building.

    From tree-loader.yaml TreeValidator specification:
    - validate() checks all aspects of a TreeDefinition
    - Returns empty list on success, list of ValidationError on failure
    - All errors include location (tree_id:node_id:line_number)
    - Suggestions provided for common typos

    Example:
        >>> validator = TreeValidator()
        >>> errors = validator.validate(tree_definition, registry)
        >>> if errors:
        ...     for error in errors:
        ...         print(error)
        ... else:
        ...     print("Validation passed!")
    """

    def __init__(
        self,
        resolve_functions: bool = True,
        check_subtrees: bool = True,
    ) -> None:
        """Initialize the validator.

        Args:
            resolve_functions: Whether to actually resolve fn paths (E4003).
                Set to False for faster validation without module imports.
            check_subtrees: Whether to check subtree references (E4004, E3002).
                Set to False if registry is not available.
        """
        self._resolve_functions = resolve_functions
        self._check_subtrees = check_subtrees

    def validate(
        self,
        definition: TreeDefinition,
        registry: Optional["TreeRegistry"] = None,
    ) -> List[ValidationError]:
        """Validate tree definition.

        From tree-loader.yaml TreeValidator.validate:

        Checks:
        - All fn paths resolve to callables (E4003)
        - All subtree_refs exist or are marked lazy (E4004)
        - No circular subtree references (E3002)
        - Required node properties present
        - Blackboard schema valid
        - Node IDs unique (E2005)
        - Node types have correct child counts (E2006)

        Args:
            definition: The tree definition to validate.
            registry: Optional TreeRegistry for subtree validation.

        Returns:
            Empty list on success, list of ValidationError on failure.
        """
        errors: List[ValidationError] = []

        # Track seen node IDs for duplicate detection
        seen_ids: Dict[str, str] = {}  # id -> location where first seen

        # Check the root node recursively
        self._validate_node(
            definition.root,
            definition.name,
            errors,
            seen_ids,
            registry,
        )

        # Check circular references in subtrees
        if self._check_subtrees and registry is not None:
            circular_errors = self._check_circular_references(
                definition,
                registry,
            )
            errors.extend(circular_errors)

        # Check blackboard schema validity
        schema_errors = self._validate_blackboard_schema(definition)
        errors.extend(schema_errors)

        return errors

    def _validate_node(
        self,
        node: NodeDefinition,
        tree_id: str,
        errors: List[ValidationError],
        seen_ids: Dict[str, str],
        registry: Optional["TreeRegistry"] = None,
    ) -> None:
        """Validate a single node and its children recursively.

        Args:
            node: Node to validate.
            tree_id: Tree ID for error locations.
            errors: List to append errors to.
            seen_ids: Set of already seen node IDs for duplicate detection.
            registry: Optional TreeRegistry for subtree validation.
        """
        location = ValidationError.make_location(
            tree_id,
            node.id,
            node.source_line or None,
        )

        # E2005: Check for duplicate node ID
        if node.id:
            if node.id in seen_ids:
                errors.append(ValidationError(
                    code="E2005",
                    location=location,
                    message=(
                        f"Duplicate node ID '{node.id}' in tree '{tree_id}'. "
                        f"First occurrence at {seen_ids[node.id]}. "
                        f"Each node must have a unique ID."
                    ),
                ))
            else:
                seen_ids[node.id] = location

            # Check ID format
            if not is_valid_node_id(node.id):
                errors.append(ValidationError(
                    code="E2004",
                    location=location,
                    message=(
                        f"Invalid node ID '{node.id}': must match pattern "
                        f"^[a-zA-Z][a-zA-Z0-9_-]*$ "
                        f"(start with letter, followed by letters/numbers/underscores/hyphens)"
                    ),
                ))

        # E2006: Check child count matches node type
        child_count = len(node.children)
        if NodeTypeEnum.is_composite(node.type):
            if child_count < 1:
                errors.append(ValidationError(
                    code="E2006",
                    location=location,
                    message=(
                        f"Node '{node.id}' is composite type '{node.type}' "
                        f"but has {child_count} children (expected: 1+)"
                    ),
                ))
        elif NodeTypeEnum.is_decorator(node.type):
            if child_count != 1:
                errors.append(ValidationError(
                    code="E2006",
                    location=location,
                    message=(
                        f"Node '{node.id}' is decorator type '{node.type}' "
                        f"but has {child_count} children (expected: 1)"
                    ),
                ))
        elif NodeTypeEnum.is_leaf(node.type):
            if child_count > 0:
                errors.append(ValidationError(
                    code="E2006",
                    location=location,
                    message=(
                        f"Node '{node.id}' is leaf type '{node.type}' "
                        f"but has {child_count} children (expected: 0)"
                    ),
                ))
        else:
            # Unknown node type
            errors.append(ValidationError(
                code="E4002",
                location=location,
                message=f"Unknown node type '{node.type}'",
            ))

        # E4003: Check function path for action nodes
        if node.type == "action":
            fn_path = node.get_fn_path()
            if fn_path:
                # Only try to resolve if flag is set
                if self._resolve_functions:
                    fn_error = self._validate_function_path(fn_path, node.id, location)
                    if fn_error:
                        errors.append(fn_error)
            else:
                # No fn path specified for action - always check this
                errors.append(ValidationError(
                    code="E4003",
                    location=location,
                    message=(
                        f"Node '{node.id}' is type 'action' but has no 'fn' "
                        f"path specified in config"
                    ),
                ))

        # Condition nodes require 'condition' expression OR 'fn' function path
        if node.type == "condition":
            condition_expr = node.config.get("condition")
            fn_path = node.config.get("fn")
            if not condition_expr and not fn_path:
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"Node '{node.id}' is type 'condition' but has no 'condition' "
                        f"expression or 'fn' function path specified in config"
                    ),
                ))

        # E4004: Check subtree reference exists
        if node.type == "subtree_ref":
            subtree_name = node.get_subtree_name()
            if subtree_name:
                # Only check if subtree exists in registry when flag is set
                if self._check_subtrees and not node.is_lazy_subtree():
                    subtree_error = self._validate_subtree_ref(
                        subtree_name,
                        node.id,
                        location,
                        registry,
                    )
                    if subtree_error:
                        errors.append(subtree_error)
            else:
                # Always check for missing tree name
                errors.append(ValidationError(
                    code="E4004",
                    location=location,
                    message=(
                        f"SubtreeRef node '{node.id}' has no 'tree' or 'name' "
                        f"specified in config"
                    ),
                ))

        # Check required properties based on node type
        prop_errors = self._validate_node_properties(node, location)
        errors.extend(prop_errors)

        # Recursively validate children
        for child in node.children:
            self._validate_node(child, tree_id, errors, seen_ids, registry)

    def _validate_function_path(
        self,
        fn_path: str,
        node_id: str,
        location: str,
    ) -> Optional[ValidationError]:
        """Validate that a function path resolves to a callable.

        Args:
            fn_path: Dotted function path like 'module.submodule.function'.
            node_id: Node ID for error message.
            location: Error location string.

        Returns:
            ValidationError if function not found, None otherwise.
        """
        try:
            # Split into module path and function name
            parts = fn_path.rsplit(".", 1)
            if len(parts) != 2:
                return ValidationError(
                    code="E4003",
                    location=location,
                    message=(
                        f"Invalid function path '{fn_path}' for node '{node_id}'. "
                        f"Expected format: 'module.submodule.function'"
                    ),
                )

            module_path, func_name = parts

            # Try to import the module
            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                return ValidationError(
                    code="E4003",
                    location=location,
                    message=(
                        f"Function '{fn_path}' not found for node '{node_id}': "
                        f"cannot import module '{module_path}' ({e})"
                    ),
                )

            # Get the function
            if not hasattr(module, func_name):
                # Look for similar function names
                available = [
                    name for name in dir(module)
                    if not name.startswith("_") and callable(getattr(module, name, None))
                ]
                suggestions = get_close_matches(func_name, available, n=1, cutoff=0.6)

                return ValidationError(
                    code="E4003",
                    location=location,
                    message=(
                        f"Function '{fn_path}' not found for node '{node_id}': "
                        f"no attribute '{func_name}' in module '{module_path}'"
                    ),
                    suggestion=suggestions[0] if suggestions else None,
                    available=available[:10] if available else None,
                )

            fn = getattr(module, func_name)

            if not callable(fn):
                return ValidationError(
                    code="E4003",
                    location=location,
                    message=(
                        f"Function '{fn_path}' for node '{node_id}' is not callable"
                    ),
                )

            return None

        except Exception as e:
            return ValidationError(
                code="E4003",
                location=location,
                message=(
                    f"Error validating function '{fn_path}' for node '{node_id}': {e}"
                ),
            )

    def _validate_subtree_ref(
        self,
        subtree_name: str,
        node_id: str,
        location: str,
        registry: Optional["TreeRegistry"],
    ) -> Optional[ValidationError]:
        """Validate that a subtree reference exists in the registry.

        Args:
            subtree_name: Name of the subtree to reference.
            node_id: Node ID for error message.
            location: Error location string.
            registry: TreeRegistry to check for subtree.

        Returns:
            ValidationError if subtree not found, None otherwise.
        """
        if registry is None:
            # No registry to check against - skip validation
            return None

        # Check if subtree exists
        available_trees = registry.list_trees()

        if subtree_name not in available_trees:
            # Check for pending loads
            pending = getattr(registry, "_pending_definitions", {})
            if subtree_name in pending:
                return None  # Will be available after current load

            # Look for similar tree names
            suggestions = get_close_matches(
                subtree_name,
                available_trees,
                n=1,
                cutoff=0.6,
            )

            return ValidationError(
                code="E4004",
                location=location,
                message=(
                    f"Subtree '{subtree_name}' referenced in node '{node_id}' not found"
                ),
                suggestion=suggestions[0] if suggestions else None,
                available=available_trees[:10] if available_trees else None,
            )

        return None

    def _check_circular_references(
        self,
        definition: TreeDefinition,
        registry: "TreeRegistry",
    ) -> List[ValidationError]:
        """Check for circular subtree references.

        Uses DFS to detect cycles in the subtree dependency graph.

        Args:
            definition: Tree definition to check.
            registry: Registry containing other tree definitions.

        Returns:
            List of ValidationError for any circular references found.
        """
        errors: List[ValidationError] = []

        # Get all subtree refs from this tree
        subtree_refs = definition.all_subtree_refs()

        if not subtree_refs:
            return errors

        # DFS to detect cycles
        for ref_name in subtree_refs:
            cycle = self._find_cycle(
                ref_name,
                registry,
                visited=set(),
                path=[definition.name],
            )

            if cycle:
                errors.append(ValidationError(
                    code="E3002",
                    location=definition.name,
                    message=(
                        f"Circular tree reference detected: "
                        f"{' -> '.join(cycle)}"
                    ),
                ))
                break  # Only report first cycle found

        return errors

    def _find_cycle(
        self,
        tree_name: str,
        registry: "TreeRegistry",
        visited: Set[str],
        path: List[str],
    ) -> Optional[List[str]]:
        """Find a cycle in the subtree dependency graph using DFS.

        Args:
            tree_name: Current tree to check.
            registry: Registry containing tree definitions.
            visited: Set of already visited trees in current path.
            path: Current path for cycle reporting.

        Returns:
            Cycle path if found, None otherwise.
        """
        if tree_name in path:
            # Found cycle
            return path + [tree_name]

        if tree_name in visited:
            # Already fully explored this subtree
            return None

        # Get the tree's subtree references
        tree = registry.get(tree_name)
        if tree is None:
            # Tree not loaded yet - can't check dependencies
            return None

        # Mark as in current path
        new_path = path + [tree_name]

        # Get subtree refs (need to access definition or scan tree)
        # For now, we'll do a simple check through the tree structure
        subtree_refs = self._extract_subtree_refs_from_tree(tree)

        for ref_name in subtree_refs:
            cycle = self._find_cycle(ref_name, registry, visited, new_path)
            if cycle:
                return cycle

        visited.add(tree_name)
        return None

    def _extract_subtree_refs_from_tree(
        self,
        tree: Any,  # BehaviorTree
    ) -> List[str]:
        """Extract subtree reference names from a built tree.

        Args:
            tree: BehaviorTree instance.

        Returns:
            List of subtree names referenced.
        """
        from ..nodes import SubtreeRef

        refs: List[str] = []

        def visit_node(node: Any) -> None:
            if isinstance(node, SubtreeRef):
                refs.append(node._tree_name)

            # Visit children
            if hasattr(node, "children"):
                for child in node.children:
                    visit_node(child)
            elif hasattr(node, "child") and node.child:
                visit_node(node.child)

        visit_node(tree.root)
        return refs

    def _validate_node_properties(
        self,
        node: NodeDefinition,
        location: str,
    ) -> List[ValidationError]:
        """Validate required properties based on node type.

        Args:
            node: Node to validate.
            location: Error location string.

        Returns:
            List of validation errors.
        """
        errors: List[ValidationError] = []
        config = node.config

        # Node-specific property requirements
        if node.type == "timeout":
            if "timeout_ms" not in config:
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"Timeout node '{node.id}' missing required 'timeout_ms' property"
                    ),
                ))
            elif not isinstance(config["timeout_ms"], (int, float)):
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"Timeout node '{node.id}' 'timeout_ms' must be a number, "
                        f"got {type(config['timeout_ms']).__name__}"
                    ),
                ))

        elif node.type == "retry":
            if "max_retries" not in config:
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"Retry node '{node.id}' missing required 'max_retries' property"
                    ),
                ))
            elif not isinstance(config["max_retries"], int):
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"Retry node '{node.id}' 'max_retries' must be an integer, "
                        f"got {type(config['max_retries']).__name__}"
                    ),
                ))

        elif node.type == "cooldown":
            if "cooldown_ms" not in config:
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"Cooldown node '{node.id}' missing required 'cooldown_ms' property"
                    ),
                ))

        elif node.type == "guard":
            if "condition" not in config:
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"Guard node '{node.id}' missing required 'condition' property"
                    ),
                ))

        elif node.type == "for_each":
            if "collection_key" not in config:
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"ForEach node '{node.id}' missing required 'collection_key' property"
                    ),
                ))
            if "item_key" not in config:
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"ForEach node '{node.id}' missing required 'item_key' property"
                    ),
                ))

        elif node.type == "script":
            if "code" not in config and "file" not in config:
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"Script node '{node.id}' must have either 'code' or 'file' property"
                    ),
                ))
            if "code" in config and "file" in config:
                errors.append(ValidationError(
                    code="E4002",
                    location=location,
                    message=(
                        f"Script node '{node.id}' cannot have both 'code' and 'file' properties"
                    ),
                ))

        elif node.type == "parallel":
            policy = config.get("policy")
            if policy == "require_n":
                if "required_successes" not in config:
                    errors.append(ValidationError(
                        code="E4002",
                        location=location,
                        message=(
                            f"Parallel node '{node.id}' with policy 'require_n' "
                            f"must specify 'required_successes'"
                        ),
                    ))

        return errors

    def _validate_blackboard_schema(
        self,
        definition: TreeDefinition,
    ) -> List[ValidationError]:
        """Validate blackboard schema entries.

        Checks that schema values are valid Pydantic model paths.

        Args:
            definition: Tree definition to validate.

        Returns:
            List of validation errors.
        """
        errors: List[ValidationError] = []

        for key, model_path in definition.blackboard_schema.items():
            # Try to resolve the model path
            try:
                parts = model_path.rsplit(".", 1)
                if len(parts) != 2:
                    errors.append(ValidationError(
                        code="E4002",
                        location=definition.name,
                        message=(
                            f"Invalid blackboard schema for key '{key}': "
                            f"'{model_path}' is not a valid model path "
                            f"(expected: 'module.Model')"
                        ),
                    ))
                    continue

                module_path, class_name = parts
                try:
                    module = importlib.import_module(module_path)
                    if not hasattr(module, class_name):
                        errors.append(ValidationError(
                            code="E4002",
                            location=definition.name,
                            message=(
                                f"Invalid blackboard schema for key '{key}': "
                                f"'{class_name}' not found in module '{module_path}'"
                            ),
                        ))
                except ImportError as e:
                    errors.append(ValidationError(
                        code="E4002",
                        location=definition.name,
                        message=(
                            f"Invalid blackboard schema for key '{key}': "
                            f"cannot import module '{module_path}' ({e})"
                        ),
                    ))

            except Exception as e:
                errors.append(ValidationError(
                    code="E4002",
                    location=definition.name,
                    message=(
                        f"Error validating blackboard schema for key '{key}': {e}"
                    ),
                ))

        return errors


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "TreeValidator",
]
