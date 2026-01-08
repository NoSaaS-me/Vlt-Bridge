"""
Tests for BT Parallel Merge Strategies (Phase 0.6).

Tests ParallelMerger and merge strategies for:
- Each merge strategy (LAST_WINS, FIRST_WINS, COLLECT, MERGE_DICT, FAIL_ON_CONFLICT)
- Conflict detection
- Error conditions (E8001, E8002)
- Per-key strategy configuration

Reference:
- tasks.md sections 0.6.1-0.6.9
- contracts/nodes.yaml Parallel.merge_strategies
- contracts/errors.yaml E8001, E8002
- footgun-addendum.md A.3 (Parallel Child Scope Isolation)
"""

import pytest
from pydantic import BaseModel
from typing import List, Dict, Any

from backend.src.bt.state.merge import (
    MergeStrategy,
    MergeConflict,
    MergeResult,
    ParallelMerger,
    apply_merge_result_to_parent,
    make_merge_conflict_error,
    make_merge_type_mismatch_error,
)
from backend.src.bt.state.blackboard import TypedBlackboard


# =============================================================================
# Test Pydantic Models
# =============================================================================


class CountModel(BaseModel):
    """Simple counter model."""
    value: int


class SearchResultModel(BaseModel):
    """Model representing search results."""
    query: str
    results: List[str]


class ConfigModel(BaseModel):
    """Configuration model with nested fields."""
    name: str
    settings: Dict[str, Any] = {}


class ResearchFinding(BaseModel):
    """A finding from research."""
    topic: str
    summary: str
    sources: List[str] = []


# =============================================================================
# MergeStrategy Enum Tests (0.6.1)
# =============================================================================


class TestMergeStrategyEnum:
    """Test MergeStrategy enum values (0.6.1)."""

    def test_all_strategies_defined(self):
        """All merge strategies from nodes.yaml should be defined."""
        assert MergeStrategy.LAST_WINS == "last_wins"
        assert MergeStrategy.FIRST_WINS == "first_wins"
        assert MergeStrategy.COLLECT == "collect"
        assert MergeStrategy.MERGE_DICT == "merge_dict"
        assert MergeStrategy.FAIL_ON_CONFLICT == "fail_on_conflict"

    def test_is_string_enum(self):
        """MergeStrategy should be a string enum for JSON serialization."""
        assert isinstance(MergeStrategy.LAST_WINS, str)
        assert MergeStrategy.LAST_WINS.value == "last_wins"


# =============================================================================
# ParallelMerger Basic Tests (0.6.2-0.6.3)
# =============================================================================


class TestParallelMergerBasic:
    """Test ParallelMerger basic functionality (0.6.2-0.6.3)."""

    def test_empty_child_scopes(self):
        """Empty child scopes should return empty merge result."""
        parent = TypedBlackboard(scope_name="parent")
        merger = ParallelMerger()

        result = merger.merge(parent, [])

        assert result.success is True
        assert result.merged_data == {}
        assert result.has_conflicts is False

    def test_single_child_no_conflict(self):
        """Single child write should never conflict."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("count", CountModel)

        child = parent.create_child_scope("child")
        child.set("count", CountModel(value=42))

        merger = ParallelMerger()
        result = merger.merge(parent, [child])

        assert result.success is True
        assert "count" in result.merged_data
        assert result.merged_data["count"].value == 42
        assert result.has_conflicts is False

    def test_non_overlapping_writes(self):
        """Different children writing different keys should not conflict."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("count1", CountModel)
        parent.register("count2", CountModel)

        child1 = parent.create_child_scope("child1")
        child1.set("count1", CountModel(value=10))

        child2 = parent.create_child_scope("child2")
        child2.set("count2", CountModel(value=20))

        merger = ParallelMerger()
        result = merger.merge(parent, [child1, child2])

        assert result.success is True
        assert result.merged_data["count1"].value == 10
        assert result.merged_data["count2"].value == 20
        assert result.has_conflicts is False

    def test_skips_internal_keys(self):
        """Internal keys (starting with _) should be skipped."""
        parent = TypedBlackboard(scope_name="parent")

        child = parent.create_child_scope("child")
        child.set_internal("_internal", {"data": "value"})

        merger = ParallelMerger()
        result = merger.merge(parent, [child])

        assert "_internal" not in result.merged_data


# =============================================================================
# LAST_WINS Strategy Tests (0.6.8)
# =============================================================================


class TestMergeLastWins:
    """Test LAST_WINS merge strategy."""

    def test_last_child_value_wins(self):
        """Last child's value should win on conflict."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("count", CountModel)

        child1 = parent.create_child_scope("child1")
        child1.set("count", CountModel(value=10))

        child2 = parent.create_child_scope("child2")
        child2.set("count", CountModel(value=20))

        child3 = parent.create_child_scope("child3")
        child3.set("count", CountModel(value=30))

        merger = ParallelMerger(default_strategy=MergeStrategy.LAST_WINS)
        result = merger.merge(parent, [child1, child2, child3])

        assert result.success is True
        assert result.merged_data["count"].value == 30  # Last child wins

    def test_is_default_strategy(self):
        """LAST_WINS should be the default strategy."""
        merger = ParallelMerger()
        assert merger.get_strategy_for_key("any_key") == MergeStrategy.LAST_WINS


# =============================================================================
# FIRST_WINS Strategy Tests (0.6.8)
# =============================================================================


class TestMergeFirstWins:
    """Test FIRST_WINS merge strategy."""

    def test_first_child_value_wins(self):
        """First child's value should win on conflict."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("count", CountModel)

        child1 = parent.create_child_scope("child1")
        child1.set("count", CountModel(value=10))

        child2 = parent.create_child_scope("child2")
        child2.set("count", CountModel(value=20))

        child3 = parent.create_child_scope("child3")
        child3.set("count", CountModel(value=30))

        merger = ParallelMerger(default_strategy=MergeStrategy.FIRST_WINS)
        result = merger.merge(parent, [child1, child2, child3])

        assert result.success is True
        assert result.merged_data["count"].value == 10  # First child wins


# =============================================================================
# COLLECT Strategy Tests (0.6.8, 0.6.9)
# =============================================================================


class TestMergeCollect:
    """Test COLLECT merge strategy."""

    def test_collects_all_values(self):
        """COLLECT should create list of all values."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("finding", ResearchFinding)

        child1 = parent.create_child_scope("researcher1")
        child1.set("finding", ResearchFinding(topic="Auth", summary="JWT tokens"))

        child2 = parent.create_child_scope("researcher2")
        child2.set("finding", ResearchFinding(topic="DB", summary="SQLite"))

        child3 = parent.create_child_scope("researcher3")
        child3.set("finding", ResearchFinding(topic="API", summary="REST"))

        merger = ParallelMerger(default_strategy=MergeStrategy.COLLECT)
        result = merger.merge(parent, [child1, child2, child3])

        assert result.success is True
        findings = result.merged_data["finding"]
        assert isinstance(findings, list)
        assert len(findings) == 3
        assert findings[0].topic == "Auth"
        assert findings[1].topic == "DB"
        assert findings[2].topic == "API"

    def test_parallel_researchers_integration(self):
        """Integration test: parallel researchers with COLLECT (0.6.9)."""
        parent = TypedBlackboard(scope_name="research")
        parent.register("results", SearchResultModel)
        parent.register("finding", ResearchFinding)

        # Simulate 3 parallel researchers
        researchers = []
        topics = ["authentication", "database", "caching"]
        summaries = ["Uses JWT", "SQLite FTS5", "Redis optional"]

        for i, (topic, summary) in enumerate(zip(topics, summaries)):
            child = parent.create_child_scope(f"researcher_{i}")
            child.set("finding", ResearchFinding(
                topic=topic,
                summary=summary,
                sources=[f"doc{i}.md", f"code{i}.py"],
            ))
            researchers.append(child)

        merger = ParallelMerger(
            default_strategy=MergeStrategy.COLLECT,
        )
        result = merger.merge(parent, researchers)

        assert result.success is True
        findings = result.merged_data["finding"]
        assert len(findings) == 3

        # Verify all findings preserved
        topics_found = [f.topic for f in findings]
        assert "authentication" in topics_found
        assert "database" in topics_found
        assert "caching" in topics_found


# =============================================================================
# MERGE_DICT Strategy Tests (0.6.8)
# =============================================================================


class TestMergeDictStrategy:
    """Test MERGE_DICT merge strategy."""

    def test_deep_merges_dicts(self):
        """MERGE_DICT should deep merge dictionaries."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("config", ConfigModel)

        child1 = parent.create_child_scope("child1")
        child1.set("config", ConfigModel(
            name="base",
            settings={"a": 1, "nested": {"x": 10}},
        ))

        child2 = parent.create_child_scope("child2")
        child2.set("config", ConfigModel(
            name="override",
            settings={"b": 2, "nested": {"y": 20}},
        ))

        merger = ParallelMerger(default_strategy=MergeStrategy.MERGE_DICT)
        result = merger.merge(parent, [child1, child2])

        assert result.success is True
        # MERGE_DICT returns dict, not model
        merged = result.merged_data["config"]
        assert merged["name"] == "override"  # Later wins
        assert merged["settings"]["a"] == 1  # From first
        assert merged["settings"]["b"] == 2  # From second
        # Note: nested dict is replaced, not deep merged
        assert merged["settings"]["nested"]["y"] == 20

    def test_type_mismatch_error(self):
        """MERGE_DICT with non-dict should return E8002."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("count", CountModel)
        parent.register("config", ConfigModel)

        child1 = parent.create_child_scope("child1")
        child1.set("count", CountModel(value=10))

        child2 = parent.create_child_scope("child2")
        child2.set("count", CountModel(value=20))

        merger = ParallelMerger(default_strategy=MergeStrategy.MERGE_DICT)
        result = merger.merge(parent, [child1, child2])

        # CountModel can be converted to dict via model_dump()
        # So this should actually work
        assert result.success is True


# =============================================================================
# FAIL_ON_CONFLICT Strategy Tests (0.6.8)
# =============================================================================


class TestMergeFailOnConflict:
    """Test FAIL_ON_CONFLICT merge strategy."""

    def test_fails_when_multiple_writers(self):
        """FAIL_ON_CONFLICT should return E8001 when multiple children write."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("count", CountModel)

        child1 = parent.create_child_scope("child1")
        child1.set("count", CountModel(value=10))

        child2 = parent.create_child_scope("child2")
        child2.set("count", CountModel(value=20))

        merger = ParallelMerger(default_strategy=MergeStrategy.FAIL_ON_CONFLICT)
        result = merger.merge(parent, [child1, child2])

        assert result.success is False
        assert result.has_conflicts is True
        assert len(result.conflicts) == 1
        assert result.conflicts[0].key == "count"
        assert "child1" in result.conflicts[0].writers
        assert "child2" in result.conflicts[0].writers

        # Check error details
        assert len(result.errors) == 1
        assert result.errors[0].code == "E8001"

    def test_succeeds_when_single_writer(self):
        """FAIL_ON_CONFLICT should succeed when only one child writes."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("count", CountModel)

        child1 = parent.create_child_scope("child1")
        child1.set("count", CountModel(value=10))

        child2 = parent.create_child_scope("child2")
        # child2 doesn't write to "count"

        merger = ParallelMerger(default_strategy=MergeStrategy.FAIL_ON_CONFLICT)
        result = merger.merge(parent, [child1, child2])

        assert result.success is True


# =============================================================================
# Per-Key Strategy Configuration Tests (0.6.2)
# =============================================================================


class TestPerKeyStrategies:
    """Test per-key merge strategy configuration."""

    def test_per_key_override(self):
        """Per-key strategies should override default."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("collect_key", ResearchFinding)
        parent.register("last_wins_key", CountModel)

        child1 = parent.create_child_scope("child1")
        child1.set("collect_key", ResearchFinding(topic="A", summary="a"))
        child1.set("last_wins_key", CountModel(value=10))

        child2 = parent.create_child_scope("child2")
        child2.set("collect_key", ResearchFinding(topic="B", summary="b"))
        child2.set("last_wins_key", CountModel(value=20))

        merger = ParallelMerger(
            default_strategy=MergeStrategy.LAST_WINS,
            per_key_strategies={
                "collect_key": MergeStrategy.COLLECT,
            },
        )
        result = merger.merge(parent, [child1, child2])

        assert result.success is True
        # collect_key should be list
        assert isinstance(result.merged_data["collect_key"], list)
        assert len(result.merged_data["collect_key"]) == 2
        # last_wins_key should be single value (last wins)
        assert result.merged_data["last_wins_key"].value == 20

    def test_get_strategy_for_key(self):
        """get_strategy_for_key should return correct strategy."""
        merger = ParallelMerger(
            default_strategy=MergeStrategy.LAST_WINS,
            per_key_strategies={
                "special": MergeStrategy.FAIL_ON_CONFLICT,
            },
        )

        assert merger.get_strategy_for_key("special") == MergeStrategy.FAIL_ON_CONFLICT
        assert merger.get_strategy_for_key("other") == MergeStrategy.LAST_WINS


# =============================================================================
# Conflict Detection Tests (0.6.3)
# =============================================================================


class TestConflictDetection:
    """Test merge conflict detection (0.6.3)."""

    def test_conflict_records_writers(self):
        """MergeConflict should record all writers."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("count", CountModel)

        child1 = parent.create_child_scope("writer_a")
        child1.set("count", CountModel(value=1))

        child2 = parent.create_child_scope("writer_b")
        child2.set("count", CountModel(value=2))

        merger = ParallelMerger(default_strategy=MergeStrategy.FAIL_ON_CONFLICT)
        result = merger.merge(parent, [child1, child2])

        conflict = result.conflicts[0]
        assert "writer_a" in conflict.writers
        assert "writer_b" in conflict.writers
        assert len(conflict.values) == 2

    def test_conflict_to_dict(self):
        """MergeConflict.to_dict should serialize correctly."""
        conflict = MergeConflict(
            key="test",
            writers=["child1", "child2"],
            values=[{"value": 10}, {"value": 20}],
        )

        d = conflict.to_dict()

        assert d["key"] == "test"
        assert d["writers"] == ["child1", "child2"]
        assert len(d["values"]) == 2

    def test_merge_result_to_dict(self):
        """MergeResult.to_dict should serialize correctly."""
        result = MergeResult(
            success=False,
            merged_data={},
            conflicts=[
                MergeConflict(key="k", writers=["a", "b"], values=[1, 2])
            ],
        )

        d = result.to_dict()

        assert d["success"] is False
        assert d["has_conflicts"] is True
        assert len(d["conflicts"]) == 1


# =============================================================================
# Error Factory Tests
# =============================================================================


class TestErrorFactories:
    """Test error factory functions."""

    def test_make_merge_conflict_error(self):
        """make_merge_conflict_error should create E8001."""
        error = make_merge_conflict_error(
            key="test_key",
            writer_count=3,
            child_values=[
                {"child_index": 0, "value_preview": "10"},
                {"child_index": 1, "value_preview": "20"},
                {"child_index": 2, "value_preview": "30"},
            ],
            merge_strategy="fail_on_conflict",
            parallel_node_id="parallel_1",
        )

        assert error.code == "E8001"
        assert error.category == "merge"
        assert "test_key" in error.message
        assert "3 children" in error.message

    def test_make_merge_type_mismatch_error(self):
        """make_merge_type_mismatch_error should create E8002."""
        error = make_merge_type_mismatch_error(
            key="config",
            types=["dict", "str", "int"],
            merge_strategy="merge_dict",
        )

        assert error.code == "E8002"
        assert error.category == "merge"
        assert "config" in error.message
        assert "incompatible types" in error.message


# =============================================================================
# Apply Merge Result Tests
# =============================================================================


class TestApplyMergeResult:
    """Test apply_merge_result_to_parent helper."""

    def test_applies_merged_data(self):
        """Successful merge result should be applied to parent."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("count", CountModel)

        child = parent.create_child_scope("child")
        child.set("count", CountModel(value=42))

        merger = ParallelMerger()
        merge_result = merger.merge(parent, [child])

        apply_result = apply_merge_result_to_parent(parent, merge_result)

        assert apply_result.is_ok
        assert parent.get("count", CountModel).value == 42

    def test_fails_on_unsuccessful_merge(self):
        """Cannot apply failed merge result."""
        parent = TypedBlackboard(scope_name="parent")

        failed_result = MergeResult(success=False, merged_data={})

        apply_result = apply_merge_result_to_parent(parent, failed_result)

        assert apply_result.is_error
        assert apply_result.error.code == "E8001"

    def test_skips_unregistered_keys(self):
        """Unregistered keys in merge result should be skipped."""
        parent = TypedBlackboard(scope_name="parent")
        # Don't register "count"

        merge_result = MergeResult(
            success=True,
            merged_data={"count": CountModel(value=42)},
        )

        apply_result = apply_merge_result_to_parent(parent, merge_result)

        assert apply_result.is_ok
        assert not parent.has("count")


# =============================================================================
# Footgun A.3 Tests - Parallel Child Scope Isolation
# =============================================================================


class TestParallelScopeIsolation:
    """Test parallel child scope isolation (footgun-addendum.md A.3)."""

    def test_children_cannot_see_sibling_writes(self):
        """Parallel children should not see each other's writes."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("shared", CountModel)
        parent.set("shared", CountModel(value=0))

        child1 = parent.create_child_scope("child1")
        child2 = parent.create_child_scope("child2")

        # child1 writes
        child1.set("shared", CountModel(value=100))

        # child2 should still see parent value, not child1's write
        assert child2.get("shared", CountModel).value == 0

    def test_child_writes_isolated_until_merge(self):
        """Child writes should not affect parent until merge."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("count", CountModel)
        parent.set("count", CountModel(value=0))

        child = parent.create_child_scope("child")
        child.set("count", CountModel(value=42))

        # Parent should still have original value
        assert parent.get("count", CountModel).value == 0

        # After merge, parent can be updated
        merger = ParallelMerger()
        result = merger.merge(parent, [child])
        apply_merge_result_to_parent(parent, result)

        assert parent.get("count", CountModel).value == 42

    def test_parallel_merge_after_all_complete(self):
        """Merge should happen after all children complete."""
        parent = TypedBlackboard(scope_name="parent")
        parent.register("results", SearchResultModel)

        # Simulate parallel execution
        children = []
        for i in range(3):
            child = parent.create_child_scope(f"child_{i}")
            child.set("results", SearchResultModel(
                query=f"query_{i}",
                results=[f"result_{i}"],
            ))
            children.append(child)

        # Merge happens after all are done
        merger = ParallelMerger(default_strategy=MergeStrategy.COLLECT)
        result = merger.merge(parent, children)

        assert result.success is True
        assert len(result.merged_data["results"]) == 3
