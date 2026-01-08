"""
BT Parallel Merge Strategies - Conflict resolution for parallel child scopes.

This module provides merge strategies for combining blackboard writes
from parallel child executions, addressing footgun A.3 from the addendum.

Part of the BT Universal Runtime (spec 019).
Implements tasks 0.6.1-0.6.9 from tasks.md.

Reference:
- contracts/nodes.yaml - Parallel node merge_strategies
- contracts/errors.yaml - E8001 (merge conflict), E8002 (type mismatch)
- footgun-addendum.md - Section A.3 (Parallel Child Scope Isolation)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel

from .base import (
    BTError,
    ErrorContext,
    ErrorResult,
    RecoveryAction,
    RecoveryInfo,
    Severity,
)
from .blackboard import TypedBlackboard


class MergeStrategy(str, Enum):
    """Strategy for merging parallel child blackboard writes.

    From contracts/nodes.yaml Parallel.merge_strategies:

    - LAST_WINS: Last child's value wins on conflict (default)
    - FIRST_WINS: First child's value wins on conflict
    - COLLECT: Collect all values into a list
    - MERGE_DICT: Deep merge dicts, later values win
    - FAIL_ON_CONFLICT: Return FAILURE if any key has multiple writers (E8001)

    The strategy determines how to resolve when multiple parallel children
    write to the same blackboard key.
    """

    LAST_WINS = "last_wins"
    FIRST_WINS = "first_wins"
    COLLECT = "collect"
    MERGE_DICT = "merge_dict"
    FAIL_ON_CONFLICT = "fail_on_conflict"


@dataclass
class MergeConflict:
    """Information about a merge conflict.

    Created when multiple parallel children write to the same key.

    Attributes:
        key: The blackboard key with conflicting writes
        writers: Names of child scopes that wrote to this key
        values: Preview of the conflicting values (truncated for logging)
    """

    key: str
    writers: List[str]
    values: List[Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Truncate value previews for logging
        value_previews = []
        for v in self.values:
            preview = str(v)[:100]
            if len(str(v)) > 100:
                preview += "..."
            value_previews.append(preview)

        return {
            "key": self.key,
            "writers": self.writers,
            "values": value_previews,
        }


@dataclass
class MergeResult:
    """Result of a parallel merge operation.

    Attributes:
        success: Whether the merge succeeded (no E8001/E8002 errors)
        merged_data: The merged data dictionary (empty on failure)
        conflicts: List of conflicts detected (for FAIL_ON_CONFLICT)
        errors: List of BTError objects if merge failed
    """

    success: bool
    merged_data: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[MergeConflict] = field(default_factory=list)
    errors: List[BTError] = field(default_factory=list)

    @property
    def has_conflicts(self) -> bool:
        """Check if any conflicts were detected."""
        return len(self.conflicts) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "merged_data": self.merged_data,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "has_conflicts": self.has_conflicts,
        }


def make_merge_conflict_error(
    key: str,
    writer_count: int,
    child_values: List[Dict[str, Any]],
    merge_strategy: str,
    parallel_node_id: Optional[str] = None,
) -> BTError:
    """Create E8001: Merge conflict error.

    From contracts/errors.yaml E8001.

    Args:
        key: Conflicting blackboard key
        writer_count: Number of children that wrote to this key
        child_values: List of {child_index, value_preview} dicts
        merge_strategy: Strategy that was attempted
        parallel_node_id: ID of the Parallel node

    Returns:
        BTError with code E8001
    """
    return BTError(
        code="E8001",
        category="merge",
        severity=Severity.ERROR,
        message=f"Parallel merge conflict on key '{key}': {writer_count} children wrote different values",
        context=ErrorContext(
            node_id=parallel_node_id,
            extra={
                "key": key,
                "writer_count": writer_count,
                "child_values": child_values,
                "merge_strategy": merge_strategy,
            },
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.ABORT,
            manual_steps="Use a different merge strategy or ensure children write to different keys",
        ),
        emit_event=True,
    )


def make_merge_type_mismatch_error(
    key: str,
    types: List[str],
    merge_strategy: str,
) -> BTError:
    """Create E8002: Merge type mismatch error.

    From contracts/errors.yaml E8002.

    Args:
        key: Key with incompatible types
        types: List of type names that conflicted
        merge_strategy: Strategy that required type compatibility

    Returns:
        BTError with code E8002
    """
    return BTError(
        code="E8002",
        category="merge",
        severity=Severity.ERROR,
        message=f"Cannot merge values for key '{key}': incompatible types ({', '.join(types)})",
        context=ErrorContext(
            extra={
                "key": key,
                "types": types,
                "merge_strategy": merge_strategy,
            },
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.ABORT,
            manual_steps="Ensure all children write same type, or use COLLECT strategy",
        ),
        emit_event=True,
    )


class ParallelMerger:
    """Merges child scope writes after parallel execution.

    Implements footgun-addendum.md A.3 pattern:
    1. Track which keys each child wrote (via get_writes())
    2. For each key written by any child:
       - If only one child wrote: use that value
       - If multiple children wrote: apply strategy
    3. Return MergeResult with merged data and any conflicts

    Example:
        >>> merger = ParallelMerger(default_strategy=MergeStrategy.LAST_WINS)
        >>> result = merger.merge(parent_bb, [child1_bb, child2_bb, child3_bb])
        >>> if result.success:
        ...     for key, value in result.merged_data.items():
        ...         parent_bb.set(key, value)

    Per-key strategies allow fine-grained control:
        >>> merger = ParallelMerger(
        ...     default_strategy=MergeStrategy.LAST_WINS,
        ...     per_key_strategies={
        ...         "search_results": MergeStrategy.COLLECT,  # Collect all results
        ...         "config": MergeStrategy.FAIL_ON_CONFLICT,  # Don't allow conflict
        ...     }
        ... )
    """

    def __init__(
        self,
        default_strategy: MergeStrategy = MergeStrategy.LAST_WINS,
        per_key_strategies: Optional[Dict[str, MergeStrategy]] = None,
    ) -> None:
        """Initialize the merger.

        Args:
            default_strategy: Strategy for keys without specific override
            per_key_strategies: Map of key -> strategy for specific keys
        """
        self._default_strategy = default_strategy
        self._per_key_strategies = per_key_strategies or {}

    def get_strategy_for_key(self, key: str) -> MergeStrategy:
        """Get the merge strategy for a specific key.

        Args:
            key: Blackboard key

        Returns:
            MergeStrategy for this key
        """
        return self._per_key_strategies.get(key, self._default_strategy)

    def merge(
        self,
        parent: TypedBlackboard,
        child_scopes: List[TypedBlackboard],
    ) -> MergeResult:
        """Merge child scope writes into merged data.

        Per footgun-addendum.md A.3:
        1. Collect all writes from all children
        2. Identify conflicts (same key written by multiple children)
        3. Apply merge strategy per key
        4. Return MergeResult

        Args:
            parent: Parent blackboard (used for schema lookup, not modified)
            child_scopes: List of child blackboards from parallel execution

        Returns:
            MergeResult with:
            - success: True if merge succeeded
            - merged_data: Dict of merged key -> value pairs
            - conflicts: List of MergeConflict objects
            - errors: List of BTError if merge failed
        """
        if not child_scopes:
            return MergeResult(success=True, merged_data={})

        # Step 1: Collect all writes from all children
        # Map: key -> list of (child_index, child_scope_name, value)
        all_writes: Dict[str, List[tuple]] = {}

        for i, child in enumerate(child_scopes):
            writes = child.get_writes()
            for key in writes:
                if key.startswith("_"):
                    # Skip internal keys (system-reserved)
                    continue

                # Get value from child's data
                if key in child._data:
                    value = child._data[key]
                    if key not in all_writes:
                        all_writes[key] = []
                    all_writes[key].append((i, child._scope_name, value))

        # Step 2 & 3: Process each key
        merged_data: Dict[str, Any] = {}
        conflicts: List[MergeConflict] = []
        errors: List[BTError] = []

        for key, writes in all_writes.items():
            strategy = self.get_strategy_for_key(key)

            if len(writes) == 1:
                # Only one child wrote - no conflict
                _, _, value = writes[0]
                merged_data[key] = value
            else:
                # Multiple writers - apply strategy
                result = self._apply_strategy(key, writes, strategy)

                if result.is_error:
                    errors.append(result.error)
                    # Create conflict record
                    conflicts.append(MergeConflict(
                        key=key,
                        writers=[w[1] for w in writes],
                        values=[w[2] for w in writes],
                    ))
                else:
                    merged_data[key] = result.value
                    # Strategy succeeded but might still record as conflict for info
                    if strategy == MergeStrategy.FAIL_ON_CONFLICT:
                        conflicts.append(MergeConflict(
                            key=key,
                            writers=[w[1] for w in writes],
                            values=[w[2] for w in writes],
                        ))

        # Build result
        success = len(errors) == 0
        return MergeResult(
            success=success,
            merged_data=merged_data if success else {},
            conflicts=conflicts,
            errors=errors,
        )

    def _apply_strategy(
        self,
        key: str,
        writes: List[tuple],
        strategy: MergeStrategy,
    ) -> ErrorResult:
        """Apply merge strategy to conflicting writes.

        Args:
            key: Blackboard key
            writes: List of (child_index, scope_name, value) tuples
            strategy: Strategy to apply

        Returns:
            ErrorResult with merged value or error
        """
        if strategy == MergeStrategy.LAST_WINS:
            return self._merge_last_wins(key, writes)
        elif strategy == MergeStrategy.FIRST_WINS:
            return self._merge_first_wins(key, writes)
        elif strategy == MergeStrategy.COLLECT:
            return self._merge_collect(key, writes)
        elif strategy == MergeStrategy.MERGE_DICT:
            return self._merge_dict(key, writes)
        elif strategy == MergeStrategy.FAIL_ON_CONFLICT:
            return self._merge_fail_on_conflict(key, writes)
        else:
            # Unknown strategy - default to last wins
            return self._merge_last_wins(key, writes)

    def _merge_last_wins(
        self,
        key: str,
        writes: List[tuple],
    ) -> ErrorResult:
        """LAST_WINS: Use the last child's value.

        Children are ordered by their index in the parallel execution.
        The last child (highest index) wins.
        """
        # writes are already ordered by child_index
        _, _, value = writes[-1]
        return ErrorResult.ok(value)

    def _merge_first_wins(
        self,
        key: str,
        writes: List[tuple],
    ) -> ErrorResult:
        """FIRST_WINS: Use the first child's value.

        The first child (lowest index) that wrote wins.
        """
        _, _, value = writes[0]
        return ErrorResult.ok(value)

    def _merge_collect(
        self,
        key: str,
        writes: List[tuple],
    ) -> ErrorResult:
        """COLLECT: Collect all values into a list.

        Creates a list containing all values in child order.
        Useful for aggregating results from parallel researchers.
        """
        values = [w[2] for w in writes]

        # Handle case where values are already lists - flatten?
        # Per spec, we create a list of values, not flatten
        return ErrorResult.ok(values)

    def _merge_dict(
        self,
        key: str,
        writes: List[tuple],
    ) -> ErrorResult:
        """MERGE_DICT: Deep merge dictionaries.

        Later values win on conflict within the dict.
        If any value is not a dict, returns E8002 error.
        """
        # Check all values are dicts or Pydantic models
        types_found: Set[str] = set()
        dict_values: List[Dict[str, Any]] = []

        for _, scope_name, value in writes:
            if isinstance(value, BaseModel):
                dict_values.append(value.model_dump())
                types_found.add(type(value).__name__)
            elif isinstance(value, dict):
                dict_values.append(value)
                types_found.add("dict")
            else:
                types_found.add(type(value).__name__)
                # Type mismatch
                error = make_merge_type_mismatch_error(
                    key=key,
                    types=list(types_found),
                    merge_strategy=MergeStrategy.MERGE_DICT.value,
                )
                return ErrorResult(success=False, error=error)

        # Deep merge all dicts
        merged = {}
        for d in dict_values:
            merged = self._deep_merge(merged, d)

        return ErrorResult.ok(merged)

    def _deep_merge(
        self,
        base: Dict[str, Any],
        overlay: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries.

        Overlay values win on conflict. Nested dicts are recursively merged.

        Args:
            base: Base dictionary
            overlay: Dictionary to merge on top

        Returns:
            New merged dictionary
        """
        result = dict(base)

        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursive merge for nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Overlay wins
                result[key] = value

        return result

    def _merge_fail_on_conflict(
        self,
        key: str,
        writes: List[tuple],
    ) -> ErrorResult:
        """FAIL_ON_CONFLICT: Return error if multiple writers.

        This is called only when len(writes) > 1, so it always fails.
        Used when parallel children must not conflict on specific keys.
        """
        child_values = [
            {"child_index": w[0], "value_preview": str(w[2])[:100]}
            for w in writes
        ]

        error = make_merge_conflict_error(
            key=key,
            writer_count=len(writes),
            child_values=child_values,
            merge_strategy=MergeStrategy.FAIL_ON_CONFLICT.value,
        )
        return ErrorResult(success=False, error=error)


def apply_merge_result_to_parent(
    parent: TypedBlackboard,
    merge_result: MergeResult,
) -> ErrorResult[None]:
    """Apply merge result to parent blackboard.

    Convenience function to write merged data to parent.
    Only call if merge_result.success is True.

    Args:
        parent: Parent blackboard to update
        merge_result: Successful merge result

    Returns:
        ErrorResult indicating success or first failure
    """
    if not merge_result.success:
        return ErrorResult.failure(
            code="E8001",
            message="Cannot apply failed merge result",
            category="merge",
        )

    for key, value in merge_result.merged_data.items():
        # Skip if key not registered
        if not parent._get_registered_schema(key):
            continue

        result = parent.set(key, value)
        if result.is_error:
            return result

    return ErrorResult.ok()


__all__ = [
    "MergeStrategy",
    "MergeConflict",
    "MergeResult",
    "ParallelMerger",
    "apply_merge_result_to_parent",
    "make_merge_conflict_error",
    "make_merge_type_mismatch_error",
]
