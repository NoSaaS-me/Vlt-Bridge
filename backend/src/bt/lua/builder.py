"""
TreeBuilder - Builds executable BehaviorTree from TreeDefinition.

Resolves all references and creates node instances. Handles:
1. Resolve all function paths for Action/Condition nodes
2. Resolve all subtree references (unless lazy)
3. Build node hierarchy with correct types
4. Create tree with blackboard
5. Validate and return

From tree-loader.yaml TreeBuilder interface.

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.7.6-2.7.10 from tasks.md
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

from .definitions import (
    NodeDefinition,
    TreeDefinition,
    NodeTypeEnum,
)

# Import node types
from ..nodes import (
    BehaviorNode,
    # Composites
    Sequence,
    Selector,
    Parallel,
    ParallelPolicy,
    ForEach,
    # Decorators
    Timeout,
    Retry,
    Guard,
    Cooldown,
    Inverter,
    AlwaysSucceed,
    AlwaysFail,
    # Leaves
    Action,
    Condition,
    SubtreeRef,
    Script,
    # Tool nodes (MCP integration)
    Tool,
    Oracle,
    CodeSearch,
    VaultSearch,
    # Errors
    FunctionNotFoundError,
)

from ..core.tree import BehaviorTree
from ..state import TypedBlackboard, MergeStrategy

if TYPE_CHECKING:
    from .registry import TreeRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Error Classes
# =============================================================================


class TreeBuildError(Exception):
    """Error during tree building.

    Wraps errors from node construction, function resolution, etc.
    """

    def __init__(
        self,
        code: str,
        message: str,
        node_id: Optional[str] = None,
        source_line: Optional[int] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.node_id = node_id
        self.source_line = source_line

        full_message = f"[{code}] {message}"
        if node_id:
            full_message += f" (node: {node_id})"
        if source_line:
            full_message += f" (line: {source_line})"

        super().__init__(full_message)


class DuplicateNodeIdError(TreeBuildError):
    """Error when duplicate node IDs are detected during build."""

    def __init__(
        self,
        node_id: str,
        tree_id: str,
        first_location: str,
        second_location: str,
    ) -> None:
        super().__init__(
            code="E2005",
            message=(
                f"Duplicate node ID '{node_id}' in tree '{tree_id}'. "
                f"First at {first_location}, second at {second_location}. "
                f"Each node must have unique ID."
            ),
            node_id=node_id,
        )


# =============================================================================
# Node Builder Registry
# =============================================================================


# Mapping from node type string to node class
NODE_BUILDERS: Dict[str, Type[BehaviorNode]] = {
    # Composites
    "sequence": Sequence,
    "selector": Selector,
    "parallel": Parallel,
    "for_each": ForEach,
    # Decorators
    "timeout": Timeout,
    "retry": Retry,
    "guard": Guard,
    "cooldown": Cooldown,
    "inverter": Inverter,
    "always_succeed": AlwaysSucceed,
    "always_fail": AlwaysFail,
    # Leaves
    "action": Action,
    "condition": Condition,
    "subtree_ref": SubtreeRef,
    "script": Script,
    # Tool nodes (MCP integration - tasks 4.1.1-4.3.8)
    "tool": Tool,
    "oracle": Oracle,
    "code_search": CodeSearch,
    "vault_search": VaultSearch,
}


# =============================================================================
# TreeBuilder
# =============================================================================


class TreeBuilder:
    """Builds executable BehaviorTree from TreeDefinition.

    From tree-loader.yaml TreeBuilder specification:
    - build() creates complete tree from definition
    - Resolves function paths and subtree references
    - Handles node-specific configuration
    - Creates tree with blackboard schema

    Example:
        >>> builder = TreeBuilder(registry)
        >>> tree = builder.build(tree_definition)
        >>> tree.tick(context)
    """

    def __init__(
        self,
        registry: Optional["TreeRegistry"] = None,
    ) -> None:
        """Initialize the tree builder.

        Args:
            registry: Optional TreeRegistry for subtree resolution.
        """
        self._registry = registry

        # Track node IDs during build for duplicate detection
        self._seen_ids: Dict[str, str] = {}  # id -> location

    def build(self, definition: TreeDefinition) -> BehaviorTree:
        """Build executable tree from definition.

        From tree-loader.yaml TreeBuilder.build:

        1. Resolve all function paths
        2. Resolve all subtree references (unless lazy)
        3. Build node hierarchy with correct types
        4. Create tree with blackboard
        5. Validate and return

        Args:
            definition: Validated TreeDefinition to build from.

        Returns:
            Executable BehaviorTree instance.

        Raises:
            TreeBuildError: If build fails.
            DuplicateNodeIdError: If duplicate node IDs detected.
            FunctionNotFoundError: If function path doesn't resolve.
        """
        # Reset tracking
        self._seen_ids = {}

        # Build root node recursively
        try:
            root = self._build_node(definition.root, definition.name)
        except TreeBuildError:
            raise
        except Exception as e:
            raise TreeBuildError(
                code="E4002",
                message=f"Failed to build tree root: {e}",
            ) from e

        # Create blackboard with schema
        blackboard = TypedBlackboard(scope_name=f"tree:{definition.name}")

        # Register blackboard schema types
        for key, model_path in definition.blackboard_schema.items():
            try:
                model_class = self._resolve_model_class(model_path)
                blackboard.register(key, model_class)
            except Exception as e:
                logger.warning(
                    f"Failed to register blackboard schema for '{key}': {e}"
                )

        # Create tree
        tree = BehaviorTree(
            id=definition.name,
            name=definition.name,
            root=root,
            description=definition.description,
            source_path=definition.source_path,
            source_hash=definition.source_hash or definition.compute_source_hash(),
            _blackboard=blackboard,
            **definition.config,
        )

        return tree

    def _build_node(
        self,
        definition: NodeDefinition,
        tree_id: str,
        parent: Optional[BehaviorNode] = None,
    ) -> BehaviorNode:
        """Recursively build node from definition.

        Args:
            definition: Node definition to build.
            tree_id: Tree ID for error messages.
            parent: Optional parent node.

        Returns:
            Built BehaviorNode instance.

        Raises:
            TreeBuildError: If node construction fails.
            DuplicateNodeIdError: If duplicate node ID detected.
        """
        node_id = definition.id
        location = f"{tree_id}:{node_id}:{definition.source_line}"

        # Check for duplicate ID (E2005)
        if node_id in self._seen_ids:
            raise DuplicateNodeIdError(
                node_id=node_id,
                tree_id=tree_id,
                first_location=self._seen_ids[node_id],
                second_location=location,
            )
        self._seen_ids[node_id] = location

        # Build children first (bottom-up construction)
        children = [
            self._build_node(child_def, tree_id)
            for child_def in definition.children
        ]

        # Get node class
        node_type = definition.type
        if node_type not in NODE_BUILDERS:
            # Try MCP stubs
            node = self._build_stub_node(definition, tree_id, children)
            if node:
                return node

            raise TreeBuildError(
                code="E4002",
                message=f"Unknown node type '{node_type}'",
                node_id=node_id,
                source_line=definition.source_line,
            )

        # Build node based on type
        try:
            if NodeTypeEnum.is_composite(node_type):
                return self._build_composite(definition, children)
            elif NodeTypeEnum.is_decorator(node_type):
                return self._build_decorator(definition, children)
            elif NodeTypeEnum.is_leaf(node_type):
                return self._build_leaf(definition, tree_id)
            else:
                raise TreeBuildError(
                    code="E4002",
                    message=f"Cannot categorize node type '{node_type}'",
                    node_id=node_id,
                )
        except TreeBuildError:
            raise
        except Exception as e:
            raise TreeBuildError(
                code="E4002",
                message=f"Failed to build node: {e}",
                node_id=node_id,
                source_line=definition.source_line,
            ) from e

    def _build_composite(
        self,
        definition: NodeDefinition,
        children: List[BehaviorNode],
    ) -> BehaviorNode:
        """Build a composite node.

        Args:
            definition: Node definition.
            children: Pre-built child nodes.

        Returns:
            Composite node instance.
        """
        node_type = definition.type
        config = definition.config

        if node_type == "sequence":
            return Sequence(
                id=definition.id,
                name=definition.name or definition.id,
                children=children,
                metadata=config.get("metadata", {}),
            )

        elif node_type == "selector":
            return Selector(
                id=definition.id,
                name=definition.name or definition.id,
                children=children,
                metadata=config.get("metadata", {}),
            )

        elif node_type == "parallel":
            # Parse policy
            policy_str = config.get("policy", "require_all")
            policy_map = {
                "require_all": ParallelPolicy.REQUIRE_ALL,
                "require_one": ParallelPolicy.REQUIRE_ONE,
                "require_n": ParallelPolicy.REQUIRE_N,
            }
            policy = policy_map.get(policy_str, ParallelPolicy.REQUIRE_ALL)

            # Parse merge strategy
            merge_str = config.get("merge_strategy", "last_wins")
            merge_map = {
                "last_wins": MergeStrategy.LAST_WINS,
                "first_wins": MergeStrategy.FIRST_WINS,
                "collect": MergeStrategy.COLLECT,
                "merge_dict": MergeStrategy.MERGE_DICT,
                "fail_on_conflict": MergeStrategy.FAIL_ON_CONFLICT,
            }
            merge_strategy = merge_map.get(merge_str, MergeStrategy.LAST_WINS)

            return Parallel(
                id=definition.id,
                name=definition.name or definition.id,
                children=children,
                policy=policy,
                required_successes=config.get("required_successes", 1),
                merge_strategy=merge_strategy,
                per_key_strategies=config.get("per_key_strategies", {}),
                metadata=config.get("metadata", {}),
            )

        elif node_type == "for_each":
            return ForEach(
                id=definition.id,
                name=definition.name or definition.id,
                children=children,
                collection_key=config["collection_key"],
                item_key=config["item_key"],
                continue_on_failure=config.get("continue_on_failure", False),
                min_items=config.get("min_items", 0),
                metadata=config.get("metadata", {}),
            )

        else:
            raise TreeBuildError(
                code="E4002",
                message=f"Unknown composite type '{node_type}'",
                node_id=definition.id,
            )

    def _build_decorator(
        self,
        definition: NodeDefinition,
        children: List[BehaviorNode],
    ) -> BehaviorNode:
        """Build a decorator node.

        Args:
            definition: Node definition.
            children: Pre-built child nodes (should be exactly 1).

        Returns:
            Decorator node instance.
        """
        if len(children) != 1:
            raise TreeBuildError(
                code="E2006",
                message=(
                    f"Decorator '{definition.type}' expects exactly 1 child, "
                    f"got {len(children)}"
                ),
                node_id=definition.id,
            )

        child = children[0]
        node_type = definition.type
        config = definition.config

        if node_type == "timeout":
            return Timeout(
                id=definition.id,
                name=definition.name or definition.id,
                child=child,
                timeout_ms=int(config["timeout_ms"]),
                metadata=config.get("metadata", {}),
            )

        elif node_type == "retry":
            return Retry(
                id=definition.id,
                name=definition.name or definition.id,
                child=child,
                max_retries=int(config["max_retries"]),
                backoff_ms=int(config.get("backoff_ms", 0)),
                metadata=config.get("metadata", {}),
            )

        elif node_type == "guard":
            return Guard(
                id=definition.id,
                name=definition.name or definition.id,
                child=child,
                condition=config["condition"],
                metadata=config.get("metadata", {}),
            )

        elif node_type == "cooldown":
            return Cooldown(
                id=definition.id,
                name=definition.name or definition.id,
                child=child,
                cooldown_ms=int(config["cooldown_ms"]),
                metadata=config.get("metadata", {}),
            )

        elif node_type == "inverter":
            return Inverter(
                id=definition.id,
                name=definition.name or definition.id,
                child=child,
                metadata=config.get("metadata", {}),
            )

        elif node_type == "always_succeed":
            return AlwaysSucceed(
                id=definition.id,
                name=definition.name or definition.id,
                child=child,
                metadata=config.get("metadata", {}),
            )

        elif node_type == "always_fail":
            return AlwaysFail(
                id=definition.id,
                name=definition.name or definition.id,
                child=child,
                metadata=config.get("metadata", {}),
            )

        else:
            raise TreeBuildError(
                code="E4002",
                message=f"Unknown decorator type '{node_type}'",
                node_id=definition.id,
            )

    def _build_leaf(
        self,
        definition: NodeDefinition,
        tree_id: str,
    ) -> BehaviorNode:
        """Build a leaf node.

        Args:
            definition: Node definition.
            tree_id: Tree ID for error context.

        Returns:
            Leaf node instance.
        """
        node_type = definition.type
        config = definition.config

        if node_type == "action":
            fn_path = config.get("fn")
            if not fn_path:
                raise TreeBuildError(
                    code="E4003",
                    message=f"Action node missing 'fn' path",
                    node_id=definition.id,
                )

            # Resolve function
            fn = self._resolve_function(fn_path, definition.id)

            return Action(
                id=definition.id,
                name=definition.name or definition.id,
                fn=fn,
                metadata=config.get("metadata", {}),
            )

        elif node_type == "condition":
            # Condition can have fn or expression
            fn_path = config.get("fn")
            expression = config.get("expression") or config.get("condition")

            if fn_path:
                fn = self._resolve_function(fn_path, definition.id)
                return Condition(
                    id=definition.id,
                    name=definition.name or definition.id,
                    condition=fn,
                    metadata=config.get("metadata", {}),
                )
            elif expression:
                return Condition(
                    id=definition.id,
                    name=definition.name or definition.id,
                    condition=expression,
                    metadata=config.get("metadata", {}),
                )
            else:
                raise TreeBuildError(
                    code="E4003",
                    message=f"Condition node needs 'fn' or 'expression'",
                    node_id=definition.id,
                )

        elif node_type == "subtree_ref":
            subtree_name = config.get("tree") or config.get("name")
            if not subtree_name:
                raise TreeBuildError(
                    code="E4004",
                    message=f"SubtreeRef node missing 'tree' or 'name'",
                    node_id=definition.id,
                )

            subtree = SubtreeRef(
                id=definition.id,
                name=definition.name or definition.id,
                tree_name=subtree_name,
                lazy=config.get("lazy", False),
                metadata=config.get("metadata", {}),
            )

            # Resolve eagerly if not lazy and registry available
            if not config.get("lazy", False) and self._registry:
                try:
                    subtree.resolve(self._registry._trees, [tree_id])
                except Exception as e:
                    logger.warning(
                        f"Failed to resolve subtree '{subtree_name}' "
                        f"for node '{definition.id}': {e}"
                    )
                    # Don't fail build - subtree might be loaded later

            return subtree

        elif node_type == "script":
            code = config.get("code")
            file = config.get("file")

            return Script(
                id=definition.id,
                name=definition.name or definition.id,
                code=code,
                file=file,
                timeout_ms=config.get("timeout_ms"),
                metadata=config.get("metadata", {}),
            )

        # Tool nodes (MCP integration - tasks 4.1.1-4.3.8)
        elif node_type == "tool":
            tool_name = config.get("tool_name")
            if not tool_name:
                raise TreeBuildError(
                    code="E4003",
                    message="Tool node missing 'tool_name'",
                    node_id=definition.id,
                )

            output = config.get("output")
            if not output:
                raise TreeBuildError(
                    code="E2001",
                    message="Tool node missing 'output' key",
                    node_id=definition.id,
                )

            return Tool(
                id=definition.id,
                name=definition.name or definition.id,
                tool_name=tool_name,
                params=config.get("params", {}),
                output=output,
                timeout_ms=config.get("timeout"),
                metadata=config.get("metadata", {}),
            )

        elif node_type == "oracle":
            question = config.get("question")
            if not question:
                raise TreeBuildError(
                    code="E2001",
                    message="Oracle node missing 'question'",
                    node_id=definition.id,
                )

            return Oracle(
                id=definition.id,
                name=definition.name or definition.id,
                question=question,
                sources=config.get("sources"),
                explain=config.get("explain", False),
                stream_to=config.get("stream_to"),
                output=config.get("output", "oracle_answer"),
                timeout_ms=config.get("timeout"),
                metadata=config.get("metadata", {}),
            )

        elif node_type == "code_search":
            operation = config.get("operation")
            if not operation:
                raise TreeBuildError(
                    code="E2001",
                    message="CodeSearch node missing 'operation'",
                    node_id=definition.id,
                )

            return CodeSearch(
                id=definition.id,
                name=definition.name or definition.id,
                operation=operation,
                query=config.get("query"),
                limit=config.get("limit", 10),
                language=config.get("language"),
                file_pattern=config.get("file_pattern"),
                scope=config.get("scope"),
                output=config.get("output", "code_results"),
                timeout_ms=config.get("timeout"),
                metadata=config.get("metadata", {}),
            )

        elif node_type == "vault_search":
            query = config.get("query")
            if not query:
                raise TreeBuildError(
                    code="E2001",
                    message="VaultSearch node missing 'query'",
                    node_id=definition.id,
                )

            return VaultSearch(
                id=definition.id,
                name=definition.name or definition.id,
                query=query,
                tags=config.get("tags"),
                limit=config.get("limit", 10),
                output=config.get("output", "notes"),
                timeout_ms=config.get("timeout"),
                metadata=config.get("metadata", {}),
            )

        else:
            # Unknown leaf type
            raise TreeBuildError(
                code="E4002",
                message=f"Unknown leaf type '{node_type}'",
                node_id=definition.id,
            )

    def _build_stub_node(
        self,
        definition: NodeDefinition,
        tree_id: str,
        children: List[BehaviorNode],
    ) -> Optional[BehaviorNode]:
        """Build a stub node for MCP leaves that may not be implemented yet.

        Stub nodes are Action nodes that log a message and return SUCCESS.

        Note: Tool, Oracle, CodeSearch, VaultSearch are now fully implemented
        in nodes/tools.py. Only llm_call remains as a stub (handled by nodes/llm.py).

        Args:
            definition: Node definition.
            tree_id: Tree ID for context.
            children: Children (should be empty for leaves).

        Returns:
            Stub Action node if applicable, None otherwise.
        """
        # Only llm_call may need a stub if LLMCallNode is not available
        # Tool, Oracle, CodeSearch, VaultSearch are now fully implemented
        stub_types = {"llm_call"}

        if definition.type not in stub_types:
            return None

        node_type = definition.type
        config = definition.config

        # Create stub function
        def stub_fn(ctx: Any) -> Any:
            from ..state import RunStatus
            logger.warning(
                f"Stub node '{definition.id}' ({node_type}) executed. "
                f"Config: {config}"
            )
            return RunStatus.SUCCESS

        # Return as Action with stub
        return Action(
            id=definition.id,
            name=definition.name or f"{node_type}-stub-{definition.id}",
            fn=stub_fn,
            metadata={
                "stub": True,
                "original_type": node_type,
                **config.get("metadata", {}),
            },
        )

    def _resolve_function(
        self,
        fn_path: str,
        node_id: str,
    ) -> Callable:
        """Resolve a function path to a callable.

        Args:
            fn_path: Dotted function path.
            node_id: Node ID for error context.

        Returns:
            Resolved callable.

        Raises:
            TreeBuildError: If function cannot be resolved.
        """
        try:
            parts = fn_path.rsplit(".", 1)
            if len(parts) != 2:
                raise TreeBuildError(
                    code="E4003",
                    message=(
                        f"Invalid function path '{fn_path}': "
                        f"expected 'module.function' format"
                    ),
                    node_id=node_id,
                )

            module_path, func_name = parts

            try:
                module = importlib.import_module(module_path)
            except ImportError as e:
                raise TreeBuildError(
                    code="E4003",
                    message=f"Cannot import module '{module_path}': {e}",
                    node_id=node_id,
                ) from e

            if not hasattr(module, func_name):
                raise TreeBuildError(
                    code="E4003",
                    message=(
                        f"Function '{func_name}' not found in module '{module_path}'"
                    ),
                    node_id=node_id,
                )

            fn = getattr(module, func_name)

            if not callable(fn):
                raise TreeBuildError(
                    code="E4003",
                    message=f"'{fn_path}' is not callable",
                    node_id=node_id,
                )

            return fn

        except TreeBuildError:
            raise
        except Exception as e:
            raise TreeBuildError(
                code="E4003",
                message=f"Error resolving function '{fn_path}': {e}",
                node_id=node_id,
            ) from e

    def _resolve_model_class(self, model_path: str) -> type:
        """Resolve a model path to a Pydantic model class.

        Args:
            model_path: Dotted model path like 'module.Model'.

        Returns:
            Model class.

        Raises:
            ImportError: If module cannot be imported.
            AttributeError: If class not found in module.
        """
        parts = model_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model path: {model_path}")

        module_path, class_name = parts
        module = importlib.import_module(module_path)

        if not hasattr(module, class_name):
            raise AttributeError(
                f"'{class_name}' not found in module '{module_path}'"
            )

        return getattr(module, class_name)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "TreeBuilder",
    "TreeBuildError",
    "DuplicateNodeIdError",
    "NODE_BUILDERS",
]
